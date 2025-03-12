import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

from longiseg.training.dataloading.longi_dataset import LongiSegBaseDataset
from longiseg.utilities.label_handling.label_handling import LabelManager


class LongiSegDataLoader(DataLoader):
    def __init__(self,
                 data: LongiSegBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = list(data.patients.keys())

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, *_ = self._data.load_single_scan(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_current_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_current_all = np.zeros(self.seg_shape, dtype=np.int16)
        data_prior_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_prior_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data_current, seg_current, data_prior, seg_prior, _, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data_current.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_current.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data_current = data_current[this_slice]
            data_prior = data_prior[this_slice]

            this_slice = tuple([slice(0, seg_current.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg_current = seg_current[this_slice]
            seg_prior = seg_prior[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_current_all[j] = np.pad(data_current, padding, 'constant', constant_values=0)
            seg_current_all[j] = np.pad(seg_current, padding, 'constant', constant_values=-1)
            data_prior_all[j] = np.pad(data_prior, padding, 'constant', constant_values=0)
            seg_prior_all[j] = np.pad(seg_prior, padding, 'constant', constant_values=-1)

        if self.patch_size_was_2d:
            data_current_all = data_current_all[:, :, 0]
            seg_current_all = seg_current_all[:, :, 0]
            data_prior_all = data_prior_all[:, :, 0]
            seg_prior_all = seg_prior_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_current_all = torch.from_numpy(data_current_all).float()
                    seg_current_all = torch.from_numpy(seg_current_all).to(torch.int16)
                    data_prior_all = torch.from_numpy(data_prior_all).float()
                    seg_prior_all = torch.from_numpy(seg_prior_all).to(torch.int16)
                    images_current = []
                    segs_current = []
                    images_prior = []
                    segs_prior = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image_current': data_current_all[b], 'segmentation_current': seg_current_all[b],
                                                 'image_prior': data_prior_all[b], 'segmentation_prior': seg_prior_all[b]})
                        images_current.append(tmp['image_current'])
                        segs_current.append(tmp['segmentation_current'])
                        images_prior.append(tmp['image_prior'])
                        segs_prior.append(tmp['segmentation_prior'])
                    data_current_all = torch.stack(images_current)
                    data_prior_all = torch.stack(images_prior)
                    if isinstance(segs_current[0], list):
                        seg_current_all = [torch.stack([s[i] for s in segs_current]) for i in range(len(segs_current[0]))]
                    else:
                        seg_current_all = torch.stack(segs_current)
                    if isinstance(segs_prior[0], list):
                        seg_prior_all = [torch.stack([s[i] for s in segs_prior]) for i in range(len(segs_prior[0]))]
                    else:
                        seg_prior_all = torch.stack(segs_prior)
                    del segs_current, images_current, segs_prior, images_prior
            return {'data_current': data_current_all, 'target_current': seg_current_all, 'data_prior': data_prior_all, 
                    'target_prior': seg_prior_all, 'keys': selected_keys}

        return {'data_current': data_current_all, 'target_current': seg_current_all, 'data_prior': data_prior_all, 
                'target_prior': seg_prior_all, 'keys': selected_keys}