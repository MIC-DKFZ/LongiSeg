import multiprocessing
import warnings
from time import sleep

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import distributed as dist
from torch import autocast

from longiseg.configuration import default_num_processes
from longiseg.evaluation.evaluate_predictions import compute_metrics_on_folder
from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor
from longiseg.inference.export_prediction import export_prediction_from_logits
from longiseg.inference.sliding_window_prediction import compute_gaussian
from longiseg.utilities.file_path_utilities import check_workers_alive_and_busy
from longiseg.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from longiseg.training.loss.dice import get_tp_fp_fn_tn
from longiseg.utilities.helpers import dummy_context

from longiseg.training.LongiSegTrainer.LongiSegTrainer import LongiSegTrainer


class LongiSegTrainerPrimed(LongiSegTrainer):
    architecture_class_name = "LongiUNetPrimed"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def train_step(self, batch: dict) -> dict:
        data_current = batch['data_current']
        target_current = batch['target_current']
        data_prior = batch['data_prior']
        target_prior = batch['target_prior']

        data_current = data_current.to(self.device, non_blocking=True)
        data_prior = data_prior.to(self.device, non_blocking=True)

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
            # every longitudinal network should take data_current, data_prior, target_prior as input, even if not all are used
            output = self.network(data_current, data_prior, target_prior)
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
        data_current = batch['data_current']
        target_current = batch['target_current']
        data_prior = batch['data_prior']
        target_prior = batch['target_prior']

        data_current = data_current.to(self.device, non_blocking=True)
        data_prior = data_prior.to(self.device, non_blocking=True)

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
            # every longitudinal network should take data_current, data_prior, target_prior as input, even if not all are used
            output = self.network(data_current, data_prior, target_prior)
            del data_current
            l = self.loss(output, target_current)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target_current[0]

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

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
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
                for j, (k_c, k_p) in enumerate(zip(dataset_val.patients[p], dataset_val.patients[p][:1] + dataset_val.patients[p][:-1])):
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                            allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                allowed_num_queued=2)

                    self.print_to_log_file(f"predicting {k_c}")
                    data_current, _, _, properties = dataset_val.load_single_scan(k_c)
                    data_prior, seg_prior, _, _ = dataset_val.load_single_scan(k_p)
    
                    data_current = data_current[:]
                    data_prior = data_prior[:]

                    if self.is_cascaded:
                        raise NotImplementedError("Cascaded is not implemented for longitudinal segmentation")

                    seg_prior = seg_prior[:]
                    seg_prior[seg_prior<0] = 0
                    data = np.vstack((data_current[:], data_prior[:], convert_labelmap_to_one_hot(seg_prior,
                                                self.label_manager.foreground_labels, output_dtype=data_current.dtype).squeeze(1)))
                    with warnings.catch_warnings():
                        # ignore 'The given NumPy array is not writable' warning
                        warnings.simplefilter("ignore")
                        data = torch.from_numpy(data[:])

                    self.print_to_log_file(f'{k_c}, shape {data.shape}, rank {self.local_rank}')
                    output_filename_truncated = join(validation_output_folder, k_c)

                    prediction = predictor.predict_sliding_window_return_logits(data)
                    prediction = prediction.cpu()

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
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()