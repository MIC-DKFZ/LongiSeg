import cc3d
import numpy as np


def get_cc(array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    cc = cc3d.connected_components(array, connectivity=connectivity)
    return cc


def get_fp_volume(gt, pred):
    if np.sum(pred) == 0:
        return 0

    pred_cc = get_cc(pred)
    overlap = pred_cc * gt
    overlapping_ccs = list(np.unique(overlap))
    fp_mask = ~np.isin(pred_cc, overlapping_ccs)
    return np.count_nonzero(fp_mask)


def get_fn_volume(gt, pred):
    if np.sum(gt) == 0:
        return np.nan

    gt_cc = get_cc(gt)
    overlap = gt_cc * pred
    overlapping_ccs = list(np.unique(overlap))
    fn_mask = ~np.isin(gt_cc, overlapping_ccs)
    return np.count_nonzero(fn_mask)


def compute_tp_fp_fn_tn(gt: np.ndarray, pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(gt, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((gt & pred) & use_mask)
    fp = np.sum(((~gt) & pred) & use_mask)
    fn = np.sum((gt & (~pred)) & use_mask)
    tn = np.sum(((~gt) & (~pred)) & use_mask)
    return tp, fp, fn, tn


def compute_volumetric_metrics(gt, pred, ignore_mask):
    if np.sum(gt) == 0:
        return np.nan, np.nan, np.nan
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred, ignore_mask)
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1
    precision = tp / (tp + fp) if tp + fp > 0 else 1
    return dice, recall, precision