import multiprocessing
import os
import warnings
from time import sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn
from torch import autocast
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from longiseg.training.loss.dice import get_tp_fp_fn_tn
from longiseg.utilities.helpers import dummy_context
from longiseg.architectures.utils import get_autopet_network_from_plans

from longiseg.configuration import default_num_processes
from longiseg.evaluation.evaluate_predictions_autopet import compute_metrics_on_folder
from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor
from longiseg.inference.export_prediction import export_prediction_from_logits
from longiseg.inference.sliding_window_prediction import compute_gaussian
from longiseg.training.dataloading.autopet_dataset import infer_dataset_class
from longiseg.training.dataloading.autopet_data_loader import AutoPETDataLoader, AutoPETDataLoaderPretrain
from longiseg.utilities.crossval_split import generate_crossval_split_longi
from longiseg.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from longiseg.utilities.file_path_utilities import check_workers_alive_and_busy
from longiseg.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from longiseg.utilities.label_handling.label_handling import determine_num_input_channels

from longiseg.training.LongiSegTrainer.LongiSegTrainer import LongiSegTrainer


class AutoPETTrainer(LongiSegTrainer):
    architecture_class_name = "LongiUNetAutoPET"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataloader = AutoPETDataLoader
        self.sigma = 1
        self.jiggle_images = 0
        self.jiggle_points = 2
        self.pretrain = False

    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.architecture_class_name,
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder, pretrain=self.pretrain)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_backbone_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_autopet_network_from_plans(
            architecture_class_name,
            arch_backbone_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = generate_crossval_split_longi(dataset.patients, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = self.dataloader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling,
                                 sigma=self.sigma, jiggle_images=self.jiggle_images,
                                 jiggle_points=self.jiggle_points)
        dl_val = self.dataloader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling,
                                  sigma=self.sigma, jiggle_images=self.jiggle_images,
                                  jiggle_points=self.jiggle_points)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        gauss_current = batch["gauss_current"]
        data_prior = batch["data_prior"]
        target_prior = batch["target_prior"]
        gauss_prior = batch["gauss_prior"]

        data_current = data_current.to(self.device, non_blocking=True)
        data_prior = data_prior.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)
        gauss_prior = gauss_prior.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)
        if isinstance(target_prior, list):
            # if we use target_prior, we only care about the highest resolution target
            target_prior = target_prior[0].to(self.device, non_blocking=True)
        else:
            target_prior = target_prior.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input
            output = self.network(data_current, data_prior, target_prior, gauss_current, gauss_prior)
            # del data
            l = self.loss(output, target_current)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        gauss_current = batch["gauss_current"]
        data_prior = batch["data_prior"]
        target_prior = batch["target_prior"]
        gauss_prior = batch["gauss_prior"]

        data_current = data_current.to(self.device, non_blocking=True)
        data_prior = data_prior.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)
        gauss_prior = gauss_prior.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)
        if isinstance(target_prior, list):
            # if we use target_prior, we only care about the highest resolution target
            target_prior = target_prior[0].to(self.device, non_blocking=True)
        else:
            target_prior = target_prior.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input, even if not all are used
            output = self.network(data_current, data_prior, target_prior, gauss_current, gauss_prior)
            del data_current
            l = self.loss(output, target_current)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target_current[0]
        else:
            target = target_current

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = LongiSegPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(2) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                raise NotImplementedError("Cascaded is not implemented for longitudinal segmentation")

            results = []

            for i, p in enumerate(dataset_val.patients):
                self.print_to_log_file(f"Predicting patient {p}")
                for fu_data, _, bl_data, bl_seg, _, properties in dataset_val.load_for_inference(p):
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                            allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                allowed_num_queued=2)

                    bl_lesion = properties['bl_lesion']
                    bl_point = [int(i) for i in properties['bl_point']]
                    fu_point = [int(i) for i in properties['fu_point']]

                    current = properties['current']
                    output_filename_truncated = join(validation_output_folder, f"{current}_{bl_lesion}")
                    if os.path.exists(output_filename_truncated + self.dataset_json['file_ending']):
                        continue

                    patch_size = self.configuration_manager.patch_size
                    fu_shape = fu_data.shape[1:]
                    bl_shape = bl_data.shape[1:]
                    dim = len(fu_shape)

                    fu_bbox_lbs = []
                    fu_bbox_ubs = []
                    bl_bbox_lbs = []
                    bl_bbox_ubs = []

                    for i in range(3):
                        if fu_point[i] >= patch_size[i] // 2 and fu_point[i] < fu_shape[i] - patch_size[i] // 2:
                            fu_lbs = fu_point[i] - patch_size[i] // 2
                        elif fu_point[i] < patch_size[i] // 2 and patch_size[i] <= fu_shape[i]:
                            fu_lbs = 0
                        elif fu_point[i] >= fu_shape[i] - patch_size[i] // 2 and patch_size[i] <= fu_shape[i]:
                            fu_lbs = fu_shape[i] - patch_size[i]
                        elif patch_size[i] > fu_shape[i]:
                            fu_lbs = -(patch_size[i] - fu_shape[i]) // 2
                        else:
                            raise RuntimeError(f"Unexpected Combination of fu_point {fu_point}, "
                                               f"patch_size {patch_size}, data_shape {fu_shape} "
                                               f"for patient {p}")
                        bl_lbs = fu_lbs + (bl_point[i] - fu_point[i])
                        fu_bbox_lbs.append(fu_lbs)
                        fu_bbox_ubs.append(fu_lbs + patch_size[i])
                        bl_bbox_lbs.append(bl_lbs)
                        bl_bbox_ubs.append(bl_lbs + patch_size[i])

                    valid_fu_bbox_lbs = np.clip(fu_bbox_lbs, a_min=0, a_max=None)
                    valid_fu_bbox_ubs = np.minimum(fu_shape, fu_bbox_ubs)
                    valid_bl_bbox_lbs = np.clip(bl_bbox_lbs, a_min=0, a_max=None)
                    valid_bl_bbox_ubs = np.minimum(bl_shape, bl_bbox_ubs)

                    fu_slice_data = tuple([slice(0, fu_data.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
                    bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
                    bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])

                    fu_data = fu_data[fu_slice_data]
                    bl_data = bl_data[bl_slice_data]
                    bl_seg = bl_seg[bl_slice_seg]

                    bl_seg = np.where(bl_seg == bl_lesion, 1, 0).astype(fu_data.dtype)

                    fu_point = [fu_point[i] - valid_fu_bbox_lbs[i] for i in range(dim)]
                    bl_point = [bl_point[i] - valid_bl_bbox_lbs[i] for i in range(dim)]

                    fu_gauss_point = self.dataloader.gauss_blob_func(fu_point, shape=fu_data.shape[1:], sigma=self.sigma)
                    bl_gauss_point = self.dataloader.gauss_blob_func(bl_point, shape=bl_data.shape[1:], sigma=self.sigma)

                    fu_padding = [(-min(0, fu_bbox_lbs[i]), max(fu_bbox_ubs[i] - fu_shape[i], 0)) for i in range(dim)]
                    fu_padding = ((0, 0), *fu_padding)
                    bl_padding = [(-min(0, bl_bbox_lbs[i]), max(bl_bbox_ubs[i] - bl_shape[i], 0)) for i in range(dim)]
                    bl_padding = ((0, 0), *bl_padding)

                    fu_data = np.pad(fu_data, fu_padding, 'constant', constant_values=0)
                    fu_gauss_point = np.pad(fu_gauss_point[None], fu_padding, 'constant', constant_values=0)
                    bl_data = np.pad(bl_data, bl_padding, 'constant', constant_values=0)
                    bl_seg = np.pad(bl_seg, bl_padding, 'constant', constant_values=0)
                    bl_gauss_point = np.pad(bl_gauss_point[None], bl_padding, 'constant', constant_values=0)

                    data = np.vstack((fu_data, bl_data, bl_seg, fu_gauss_point, bl_gauss_point))

                    with warnings.catch_warnings():
                        # ignore 'The given NumPy array is not writable' warning
                        warnings.simplefilter("ignore")
                        data = torch.from_numpy(data[:])

                    predicted_patch = predictor.predict_sliding_window_return_logits(data)
                    predicted_patch = predicted_patch.cpu().numpy()

                    patch_crop_slice = tuple([slice(fu_padding[i][0], predicted_patch.shape[i] - fu_padding[i][1]) for i in range(dim + 1)])
                    predicted_patch = predicted_patch[patch_crop_slice]

                    prediction = np.zeros((predicted_patch.shape[0], *fu_shape), dtype=predicted_patch.dtype)
                    prediction[0] = np.max(predicted_patch)
                    prediction[1] = np.min(predicted_patch)

                    prediction_slice = tuple([slice(0, prediction.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
                    prediction[prediction_slice] = predicted_patch

                    # this needs to go into background processes
                    results.append(
                        segmentation_export_pool.starmap_async(
                            export_prediction_from_logits, (
                                (prediction, properties, self.configuration_manager, self.plans_manager,
                                self.dataset_json, output_filename_truncated, save_probabilities),
                            )
                        )
                    )
                    # for debug purposes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #      self.dataset_json, output_filename_truncated, save_probabilities)

                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 4 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'longi_summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


class AutoPETTrainerCrossSecPoint(AutoPETTrainer):
    architecture_class_name = "CrossSecUNetPointAutoPET"

    def train_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        gauss_current = batch["gauss_current"]

        data_current = data_current.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input
            output = self.network(data_current, None, None, gauss_current, None)
            # del data
            l = self.loss(output, target_current)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        gauss_current = batch["gauss_current"]

        data_current = data_current.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input, even if not all are used
            output = self.network(data_current, None, None, gauss_current, None)
            del data_current
            l = self.loss(output, target_current)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target_current[0]
        else:
            target = target_current

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class AutoPETTrainerCrossSecMask(AutoPETTrainerCrossSecPoint):
    architecture_class_name = "CrossSecUNetMaskAutoPET"

    def train_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        target_prior = batch["target_prior"]
        gauss_current = batch["gauss_current"]

        data_current = data_current.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)
        if isinstance(target_prior, list):
            # if we use target_prior, we only care about the highest resolution target
            target_prior = target_prior[0].to(self.device, non_blocking=True)
        else:
            target_prior = target_prior.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input
            output = self.network(data_current, None, target_prior, gauss_current, None)
            # del data
            l = self.loss(output, target_current)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data_current = batch["data_current"]
        target_current = batch["target_current"]
        target_prior = batch["target_prior"]
        gauss_current = batch["gauss_current"]

        data_current = data_current.to(self.device, non_blocking=True)
        gauss_current = gauss_current.to(self.device, non_blocking=True)

        if isinstance(target_current, list):
            target_current = [i.to(self.device, non_blocking=True) for i in target_current]
        else:
            target_current = target_current.to(self.device, non_blocking=True)
        if isinstance(target_prior, list):
            # if we use target_prior, we only care about the highest resolution target
            target_prior = target_prior[0].to(self.device, non_blocking=True)
        else:
            target_prior = target_prior.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # every longitudinal network should take data_current, data_prior as input, even if not all are used
            output = self.network(data_current, None, target_prior, gauss_current, None)
            del data_current
            l = self.loss(output, target_current)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target_current[0]
        else:
            target = target_current

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class AutoPETTrainerPretrain(AutoPETTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataloader = AutoPETDataLoaderPretrain
        self.pretrain = True

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder, pretrain=self.pretrain)

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                        identifiers=None,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        # if the split file does not exist we need to create it
        if not isfile(splits_file):
            self.print_to_log_file("Creating split with 5% validation")
            splits = generate_crossval_split_longi(dataset.patients, seed=12345, n_splits=20)[:1]
            save_json(splits, splits_file)
        else:
            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_json(splits_file)
            if len(splits) > 1:
                self.print_to_log_file("WARNING: The split file contains more than one split. "
                                        "This is not intended for pretraining purposes!")

        if self.fold != 0:
            self.print_to_log_file(f"WARNING: fold {self.fold} not supported for pretraining, switching to fold 0 instead")
            self.fold = 0

        self.print_to_log_file("Using fold 0 for training per default.")
        tr_keys = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']
        self.print_to_log_file("This split has %d training and %d validation cases."
                                % (len(tr_keys), len(val_keys)))
        if any([i in val_keys for i in tr_keys]):
            self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass


class AutoPETTrainerCrossSecMaskPretrain(AutoPETTrainerCrossSecMask):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dataloader = AutoPETDataLoaderPretrain
        self.pretrain = True

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder, pretrain=self.pretrain)

        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                        identifiers=None,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        # if the split file does not exist we need to create it
        if not isfile(splits_file):
            self.print_to_log_file("Creating split with 5% validation")
            splits = generate_crossval_split_longi(dataset.patients, seed=12345, n_splits=20)[:1]
            save_json(splits, splits_file)
        else:
            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_json(splits_file)
            if len(splits) > 1:
                self.print_to_log_file("WARNING: The split file contains more than one split. "
                                        "This is not intended for pretraining purposes!")

        if self.fold != 0:
            self.print_to_log_file(f"WARNING: fold {self.fold} not supported for pretraining, switching to fold 0 instead")
            self.fold = 0

        self.print_to_log_file("Using fold 0 for training per default.")
        tr_keys = splits[self.fold]['train']
        val_keys = splits[self.fold]['val']
        self.print_to_log_file("This split has %d training and %d validation cases."
                                % (len(tr_keys), len(val_keys)))
        if any([i in val_keys for i in tr_keys]):
            self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass