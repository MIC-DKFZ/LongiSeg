from typing import List, Sequence, Tuple

import numpy as np


def compute_paired_patch_bboxes(fu_shape: Sequence[int], fu_point: Sequence[int],
                                bl_shape: Sequence[int], bl_point: Sequence[int],
                                patch_size: Sequence[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    dim = len(fu_shape)
    fu_bbox_lbs, fu_bbox_ubs, bl_bbox_lbs, bl_bbox_ubs = [], [], [], []

    for d in range(dim):
        if patch_size[d] // 2 <= fu_point[d] < fu_shape[d] - patch_size[d] // 2:
            fu_lbs = fu_point[d] - patch_size[d] // 2
        elif fu_point[d] < patch_size[d] // 2 and patch_size[d] <= fu_shape[d]:
            fu_lbs = 0
        elif fu_point[d] >= fu_shape[d] - patch_size[d] // 2 and patch_size[d] <= fu_shape[d]:
            fu_lbs = fu_shape[d] - patch_size[d]
        elif patch_size[d] > fu_shape[d]:
            fu_lbs = -(patch_size[d] - fu_shape[d]) // 2
        else:
            raise RuntimeError(f"Unexpected combination of fu_point {list(fu_point)}, patch_size {list(patch_size)} "
                               f"and data_shape {list(fu_shape)}")
        fu_bbox_lbs.append(fu_lbs)
        fu_bbox_ubs.append(fu_lbs + patch_size[d])
        bl_lbs = fu_lbs + (bl_point[d] - fu_point[d])
        bl_bbox_lbs.append(bl_lbs)
        bl_bbox_ubs.append(bl_lbs + patch_size[d])

    return fu_bbox_lbs, fu_bbox_ubs, bl_bbox_lbs, bl_bbox_ubs


def crop_bbox_to_shape(bbox_lbs: Sequence[int], bbox_ubs: Sequence[int],
                       shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    valid_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
    valid_ubs = np.minimum(shape, bbox_ubs)
    padding = [(-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0)) for d in range(len(shape))]
    return valid_lbs, valid_ubs, padding
