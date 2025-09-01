from typing import Union, Tuple, List

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from longiseg.training.dataloading.longi_dataset import LongiSegBaseDataset
from longiseg.utilities.label_handling.label_handling import LabelManager
from longiseg.training.dataloading.longi_data_loader import LongiSegDataLoader
from longiseg.training.dataloading.utils import generated_sparse_to_dense_point_rescaled_gauss


class AutoPETDataLoader(LongiSegDataLoader):
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
                 transforms=None,
                 sigma: float = 1.0,
                 jiggle_images: int = 0,
                 jiggle_points: int = 2):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.sigma = sigma
        self.jiggle_images = jiggle_images
        self.jiggle_points = jiggle_points

    @staticmethod
    def gauss_blob_func(*args, **kwargs):
        return generated_sparse_to_dense_point_rescaled_gauss(*args, **kwargs)

    def get_bbox(self, fu_shape: np.ndarray, fu_point: List[float], bl_shape: np.ndarray, bl_point: List[float], verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad_fu = self.need_to_pad.copy()
        need_to_pad_bl = self.need_to_pad.copy()
        dim = len(fu_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad_fu[d] + fu_shape[d] < self.patch_size[d]:
                need_to_pad_fu[d] = self.patch_size[d] - fu_shape[d]
            if need_to_pad_bl[d] + bl_shape[d] < self.patch_size[d]:
                need_to_pad_bl[d] = self.patch_size[d] - bl_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs_fu = [- need_to_pad_fu[i] // 2 for i in range(dim)]
        ubs_fu = [fu_shape[i] + need_to_pad_fu[i] // 2 + need_to_pad_fu[i] % 2 for i in range(dim)]

        center = np.random.randint(
            [max(lbs_fu[i] + self.patch_size[i] // 2, min(fu_point[i] - self.final_patch_size[i] // 4, 
                                                            ubs_fu[i] - self.patch_size[i] // 2)) for i in range(dim)],
            [min(ubs_fu[i] + 1 - self.patch_size[i] // 2, max(fu_point[i] + 1 + self.final_patch_size[i] // 4, 
                                                            lbs_fu[i] + 1 + self.patch_size[i] // 2)) for i in range(dim)]
        ).tolist()

        bbox_lbs_fu = [max(lbs_fu[i], center[i] - self.patch_size[i] // 2) for i in range(dim)]
        bbox_ubs_fu = [bbox_lbs_fu[i] + self.patch_size[i] for i in range(dim)]

        fu_point_relative = [fu_point[i] - bbox_lbs_fu[i] for i in range(dim)]

        shift_bl_bbox = [fu_point_relative[i] + np.random.randint(-self.jiggle_images, self.jiggle_images + 1) for i in range(dim)] 
        shift_bl_bbox = [max(0, min(shift_bl_bbox[i], self.patch_size[i] - 1)) for i in range(dim)]
        bbox_lbs_bl = [bl_point[i] - shift_bl_bbox[i] for i in range(dim)]
        bbox_ubs_bl = [bbox_lbs_bl[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs_fu, bbox_ubs_fu, bbox_lbs_bl, bbox_ubs_bl

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        fu_data_all = np.zeros(self.data_shape, dtype=np.float32)
        fu_seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        fu_gauss_point_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        bl_data_all = np.zeros(self.data_shape, dtype=np.float32)
        bl_seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        bl_gauss_point_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            fu_data, fu_seg, bl_data, bl_seg, _, properties = self._data.load_case(i)

            fu_lesion = properties['fu_lesion']
            all_fu_lesions = properties['all_fu_lesions']
            fu_point = properties['fu_point']
            bl_lesion = properties['bl_lesion']
            bl_point = properties['bl_point']

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            fu_shape = fu_data.shape[1:]
            bl_shape = bl_data.shape[1:]
            dim = len(fu_shape)

            fu_point = np.clip([int(p + self.jiggle_points * np.random.uniform(-2, 2)) for p in fu_point], a_min=0, a_max=np.array(fu_shape) - 1).tolist()
            bl_point = np.clip([int(p + self.jiggle_points * np.random.uniform(-2, 2)) for p in bl_point], a_min=0, a_max=np.array(bl_shape) - 1).tolist()

            fu_bbox_lbs, fu_bbox_ubs, bl_bbox_lbs, bl_bbox_ubs = self.get_bbox(fu_shape, fu_point, bl_shape, bl_point)

            valid_fu_bbox_lbs = np.clip(fu_bbox_lbs, a_min=0, a_max=None)
            valid_fu_bbox_ubs = np.minimum(fu_shape, fu_bbox_ubs)
            valid_bl_bbox_lbs = np.clip(bl_bbox_lbs, a_min=0, a_max=None)
            valid_bl_bbox_ubs = np.minimum(bl_shape, bl_bbox_ubs)

            fu_slice_data = tuple([slice(0, fu_data.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
            fu_slice_seg = tuple([slice(0, fu_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
            bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
            bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])

            fu_data = fu_data[fu_slice_data]
            fu_seg = fu_seg[fu_slice_seg]
            bl_data = bl_data[bl_slice_data]
            bl_seg = bl_seg[bl_slice_seg]

            fu_point = [fu_point[i] - valid_fu_bbox_lbs[i] for i in range(dim)]
            bl_point = [bl_point[i] - valid_bl_bbox_lbs[i] for i in range(dim)]

            fu_gauss_point = self.gauss_blob_func(fu_point, shape=fu_data.shape[1:], sigma=self.sigma)
            bl_gauss_point = self.gauss_blob_func(bl_point, shape=bl_data.shape[1:], sigma=self.sigma)

            if fu_lesion == 0:
                fu_seg = np.where(fu_seg < 0, -1, 0)
            else:
                fu_seg = np.select([np.isin(fu_seg, all_fu_lesions), fu_seg < 0],
                                   [1, -1], default=0)
            if bl_lesion == 0:
                bl_seg = np.where(bl_seg < 0, -1, 0)
            else:
                bl_seg = np.select([bl_seg == bl_lesion, bl_seg < 0], [1, -1], default=0)

            fu_padding = [(-min(0, fu_bbox_lbs[i]), max(fu_bbox_ubs[i] - fu_shape[i], 0)) for i in range(dim)]
            fu_padding = ((0, 0), *fu_padding)
            bl_padding = [(-min(0, bl_bbox_lbs[i]), max(bl_bbox_ubs[i] - bl_shape[i], 0)) for i in range(dim)]
            bl_padding = ((0, 0), *bl_padding)

            fu_data_all[j] = np.pad(fu_data, fu_padding, 'constant', constant_values=0)
            fu_seg_all[j] = np.pad(fu_seg, fu_padding, 'constant', constant_values=-1)
            fu_gauss_point_all[j] = np.pad(fu_gauss_point[None], fu_padding, 'constant', constant_values=0)
            bl_data_all[j] = np.pad(bl_data, bl_padding, 'constant', constant_values=0)
            bl_seg_all[j] = np.pad(bl_seg, bl_padding, 'constant', constant_values=-1)
            bl_gauss_point_all[j] = np.pad(bl_gauss_point[None], bl_padding, 'constant', constant_values=0)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    fu_data_all = torch.from_numpy(fu_data_all).float()
                    fu_seg_all = torch.from_numpy(fu_seg_all).to(torch.int16)
                    fu_gauss_point_all = torch.from_numpy(fu_gauss_point_all).float()
                    bl_data_all = torch.from_numpy(bl_data_all).float()
                    bl_seg_all = torch.from_numpy(bl_seg_all).to(torch.int16)
                    bl_gauss_point_all = torch.from_numpy(bl_gauss_point_all).float()
                    images_current = []
                    segs_current = []
                    gauss_current = []
                    images_prior = []
                    segs_prior = []
                    gauss_prior = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image_current': fu_data_all[b], 'segmentation_current': fu_seg_all[b], "gauss_current": fu_gauss_point_all[b],
                                                 'image_prior': bl_data_all[b], 'segmentation_prior': bl_seg_all[b], "gauss_prior": bl_gauss_point_all[b]})
                        images_current.append(tmp['image_current'])
                        segs_current.append(tmp['segmentation_current'])
                        gauss_current.append(tmp['gauss_current'])
                        images_prior.append(tmp['image_prior'])
                        segs_prior.append(tmp['segmentation_prior'])
                        gauss_prior.append(tmp['gauss_prior'])
                    fu_data_all = torch.stack(images_current)
                    bl_data_all = torch.stack(images_prior)
                    if isinstance(segs_current[0], list):
                        fu_seg_all = [torch.stack([s[i] for s in segs_current]) for i in range(len(segs_current[0]))]
                    else:
                        fu_seg_all = torch.stack(segs_current)
                    if isinstance(segs_prior[0], list):
                        bl_seg_all = [torch.stack([s[i] for s in segs_prior]) for i in range(len(segs_prior[0]))]
                    else:
                        bl_seg_all = torch.stack(segs_prior)
                    fu_gauss_point_all = torch.stack(gauss_current)
                    bl_gauss_point_all = torch.stack(gauss_prior)
                    del gauss_current, segs_current, images_current, gauss_prior, segs_prior, images_prior
            return {'data_current': fu_data_all, 'target_current': fu_seg_all, "gauss_current": fu_gauss_point_all,
                    'data_prior': bl_data_all, 'target_prior': bl_seg_all, "gauss_prior": bl_gauss_point_all,
                    'keys': selected_keys}

        return {'data_current': fu_data_all, 'target_current': fu_seg_all, "gauss_current": fu_gauss_point_all,
                'data_prior': bl_data_all, 'target_prior': bl_seg_all, "gauss_prior": bl_gauss_point_all,
                'keys': selected_keys}


class AutoPETDataLoaderPretrain(AutoPETDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        fu_data_all = np.zeros(self.data_shape, dtype=np.float32)
        fu_seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        fu_gauss_point_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        bl_data_all = np.zeros(self.data_shape, dtype=np.float32)
        bl_seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        bl_gauss_point_all = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            fu_data, fu_seg, bl_data, bl_seg, _, fu_properties, bl_properties = self._data.load_case(i)

            fu_lesions = list(fu_properties['class_locations'].keys())
            bl_lesions = list(bl_properties['class_locations'].keys())
            all_lesions = list(set(fu_lesions) & set(bl_lesions))

            while len(all_lesions) > 0:
                lesion = np.random.choice(all_lesions)
                if fu_properties['class_locations'][lesion]['coords'].size > 0 and bl_properties['class_locations'][lesion]['coords'].size > 0:
                    break
                all_lesions.remove(lesion)
            if len(all_lesions) == 0:
                raise RuntimeError(f"Patient {i} has no lesions with coordinates in both follow-up and baseline data.")

            fu_coords = fu_properties['class_locations'][lesion]['coords']
            fu_edt = fu_properties['class_locations'][lesion]['edt_values']
            fu_idx = np.random.choice(len(fu_coords), p=fu_edt**2 / np.sum(fu_edt**2))
            fu_point = fu_coords[fu_idx][1:]

            bl_coords = bl_properties['class_locations'][lesion]['coords']
            bl_edt = bl_properties['class_locations'][lesion]['edt_values']
            bl_idx = np.random.choice(len(bl_coords), p=bl_edt**2 / np.sum(bl_edt**2))
            bl_point = bl_coords[bl_idx][1:]

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            fu_shape = fu_data.shape[1:]
            bl_shape = bl_data.shape[1:]
            dim = len(fu_shape)

            fu_point = np.clip([int(p + self.jiggle_points * np.random.uniform(-2, 2)) for p in fu_point], a_min=0, a_max=np.array(fu_shape) - 1).tolist()
            bl_point = np.clip([int(p + self.jiggle_points * np.random.uniform(-2, 2)) for p in bl_point], a_min=0, a_max=np.array(bl_shape) - 1).tolist()

            fu_bbox_lbs, fu_bbox_ubs, bl_bbox_lbs, bl_bbox_ubs = self.get_bbox(fu_shape, fu_point, bl_shape, bl_point)

            valid_fu_bbox_lbs = np.clip(fu_bbox_lbs, a_min=0, a_max=None)
            valid_fu_bbox_ubs = np.minimum(fu_shape, fu_bbox_ubs)
            valid_bl_bbox_lbs = np.clip(bl_bbox_lbs, a_min=0, a_max=None)
            valid_bl_bbox_ubs = np.minimum(bl_shape, bl_bbox_ubs)

            fu_slice_data = tuple([slice(0, fu_data.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
            fu_slice_seg = tuple([slice(0, fu_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
            bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
            bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])

            fu_data = fu_data[fu_slice_data]
            fu_seg = fu_seg[fu_slice_seg]
            bl_data = bl_data[bl_slice_data]
            bl_seg = bl_seg[bl_slice_seg]

            fu_point = [fu_point[i] - valid_fu_bbox_lbs[i] for i in range(dim)]
            bl_point = [bl_point[i] - valid_bl_bbox_lbs[i] for i in range(dim)]

            fu_gauss_point = self.gauss_blob_func(fu_point, shape=fu_data.shape[1:], sigma=self.sigma)
            bl_gauss_point = self.gauss_blob_func(bl_point, shape=bl_data.shape[1:], sigma=self.sigma)

            fu_seg = np.select([fu_seg == int(lesion), fu_seg < 0], [1, -1], default=0)
            bl_seg = np.select([bl_seg == int(lesion), bl_seg < 0], [1, -1], default=0)

            fu_padding = [(-min(0, fu_bbox_lbs[i]), max(fu_bbox_ubs[i] - fu_shape[i], 0)) for i in range(dim)]
            fu_padding = ((0, 0), *fu_padding)
            bl_padding = [(-min(0, bl_bbox_lbs[i]), max(bl_bbox_ubs[i] - bl_shape[i], 0)) for i in range(dim)]
            bl_padding = ((0, 0), *bl_padding)

            fu_data_all[j] = np.pad(fu_data, fu_padding, 'constant', constant_values=0)
            fu_seg_all[j] = np.pad(fu_seg, fu_padding, 'constant', constant_values=-1)
            fu_gauss_point_all[j] = np.pad(fu_gauss_point[None], fu_padding, 'constant', constant_values=0)
            bl_data_all[j] = np.pad(bl_data, bl_padding, 'constant', constant_values=0)
            bl_seg_all[j] = np.pad(bl_seg, bl_padding, 'constant', constant_values=-1)
            bl_gauss_point_all[j] = np.pad(bl_gauss_point[None], bl_padding, 'constant', constant_values=0)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    fu_data_all = torch.from_numpy(fu_data_all).float()
                    fu_seg_all = torch.from_numpy(fu_seg_all).to(torch.int16)
                    fu_gauss_point_all = torch.from_numpy(fu_gauss_point_all).float()
                    bl_data_all = torch.from_numpy(bl_data_all).float()
                    bl_seg_all = torch.from_numpy(bl_seg_all).to(torch.int16)
                    bl_gauss_point_all = torch.from_numpy(bl_gauss_point_all).float()
                    images_current = []
                    segs_current = []
                    gauss_current = []
                    images_prior = []
                    segs_prior = []
                    gauss_prior = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image_current': fu_data_all[b], 'segmentation_current': fu_seg_all[b], "gauss_current": fu_gauss_point_all[b],
                                                 'image_prior': bl_data_all[b], 'segmentation_prior': bl_seg_all[b], "gauss_prior": bl_gauss_point_all[b]})
                        images_current.append(tmp['image_current'])
                        segs_current.append(tmp['segmentation_current'])
                        gauss_current.append(tmp['gauss_current'])
                        images_prior.append(tmp['image_prior'])
                        segs_prior.append(tmp['segmentation_prior'])
                        gauss_prior.append(tmp['gauss_prior'])
                    fu_data_all = torch.stack(images_current)
                    bl_data_all = torch.stack(images_prior)
                    if isinstance(segs_current[0], list):
                        fu_seg_all = [torch.stack([s[i] for s in segs_current]) for i in range(len(segs_current[0]))]
                    else:
                        fu_seg_all = torch.stack(segs_current)
                    if isinstance(segs_prior[0], list):
                        bl_seg_all = [torch.stack([s[i] for s in segs_prior]) for i in range(len(segs_prior[0]))]
                    else:
                        bl_seg_all = torch.stack(segs_prior)
                    fu_gauss_point_all = torch.stack(gauss_current)
                    bl_gauss_point_all = torch.stack(gauss_prior)
                    del gauss_current, segs_current, images_current, gauss_prior, segs_prior, images_prior
            return {'data_current': fu_data_all, 'target_current': fu_seg_all, "gauss_current": fu_gauss_point_all,
                    'data_prior': bl_data_all, 'target_prior': bl_seg_all, "gauss_prior": bl_gauss_point_all,
                    'keys': selected_keys}

        return {'data_current': fu_data_all, 'target_current': fu_seg_all, "gauss_current": fu_gauss_point_all,
                'data_prior': bl_data_all, 'target_prior': bl_seg_all, "gauss_prior": bl_gauss_point_all,
                'keys': selected_keys}