from typing import Optional
import numpy as np
from scipy.spatial import KDTree
from scipy import ndimage


def SSD(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float] = (1., 1., 1.), ignore_mask: Optional[np.ndarray] = None) -> float:
    # slightly adapted from https://github.com/emrekavur/CHAOS-evaluation/blob/master/Python/CHAOSmetrics.py
    gt_mask = gt if ignore_mask is None else gt & ~ignore_mask
    pred_mask = pred if ignore_mask is None else pred & ~ignore_mask
    gt_sum = np.sum(gt_mask)
    pred_sum = np.sum(pred_mask)
    if gt_sum == 0 or pred_sum == 0:
        return (0, 0) if gt_sum == pred_sum else (1000, 1000) # maximum value chosen to fit the largest volumes

    struct = ndimage.generate_binary_structure(3, 1)
    spacing = np.array(spacing)

    gt_border = gt_mask ^ ndimage.binary_erosion(gt_mask, structure=struct, border_value=1)
    gt_border_voxels = np.array(np.where(gt_border)).T * spacing

    pred_border = pred_mask ^ ndimage.binary_erosion(pred_mask, structure=struct, border_value=1)
    pred_border_voxels = np.array(np.where(pred_border)).T * spacing

    tree_ref = KDTree(gt_border_voxels)
    dist_seg_to_ref, _ = tree_ref.query(pred_border_voxels)
    tree_seg = KDTree(pred_border_voxels)
    dist_ref_to_seg, _ = tree_seg.query(gt_border_voxels)

    assd = (dist_seg_to_ref.sum() + dist_ref_to_seg.sum()) / (len(dist_seg_to_ref) + len(dist_ref_to_seg))
    hd95 = np.percentile(np.concatenate((dist_seg_to_ref, dist_ref_to_seg)), 95)
    return assd, hd95