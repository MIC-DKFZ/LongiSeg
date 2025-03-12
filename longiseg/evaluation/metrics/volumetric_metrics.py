import numpy as np


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_volumetric_metrics(mask_ref, mask_pred, ignore_mask):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1
    recall = tp / (tp + fn) if tp + fn > 0 else 1
    precision = tp / (tp + fp) if tp + fp > 0 else 1
    return dice, iou, recall, precision, tp, fp, fn, tn