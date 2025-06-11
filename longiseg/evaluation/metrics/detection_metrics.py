from typing import Optional
import numpy as np
import cc3d
from skimage.morphology import dilation
from scipy import sparse


def get_instances(gt: np.ndarray, pred: np.ndarray, footprint: int = 0, spacing: tuple[float] = (1., 1., 1.)):
    if footprint:
        x, y, z = np.ceil(np.divide(footprint, spacing)).astype(int)
        struct = np.ones((x, y, z), dtype=np.uint8)
        dilated_gt = dilation(gt.astype(np.uint8), struct)
        dilated_pred = dilation(pred.astype(np.uint8), struct)
    else:
        dilated_gt = gt.astype(np.uint8)
        dilated_pred = pred.astype(np.uint8)
    gt_inst, gt_inst_num = cc3d.connected_components(dilated_gt, return_N=True)
    pred_inst, pred_inst_num = cc3d.connected_components(dilated_pred, return_N=True)
    gt_inst[(gt!=1)] = 0
    pred_inst[(pred!=1)] = 0
    return gt_inst, gt_inst_num, pred_inst, pred_inst_num


def get_inst_TPFPFN(gt_inst, gt_inst_num, pred_inst, pred_inst_num):
    M, N = gt_inst_num, pred_inst_num
    if M == 0 or N == 0:
        return 0, 0, M, N

    gt_inst_flat = gt_inst.flatten()
    pred_inst_flat = pred_inst.flatten()

    gt_masks = gt_inst_flat==np.arange(1, M+1)[:, None]
    pred_masks = pred_inst_flat==np.arange(1, N+1)[:, None]

    gt_size = np.bincount(gt_inst_flat)
    pred_size = np.bincount(pred_inst_flat)

    gt_masks_sparse = [sparse.coo_matrix(mask) for mask in gt_masks]
    pred_masks_sparse = [sparse.coo_matrix(mask) for mask in pred_masks]

    iou_data = []
    rows, cols = [], []
    for i in range(M):
        for j in range(N):
            # Efficient intersection computation for sparse matrices
            gt_mask, pred_mask = gt_masks_sparse[i], pred_masks_sparse[j]
            intersection = len(set(gt_mask.col) & set(pred_mask.col))
            if intersection > 0:
                union = gt_size[i + 1] + pred_size[j + 1] - intersection
                iou_data.append(intersection / union)
                rows.append(i)
                cols.append(j)

    iou_data_gt = np.array(iou_data)
    rows_gt = np.array(rows)
    cols_gt = np.array(cols)
    iou_data_pred = np.array(iou_data)
    rows_pred = np.array(rows)
    cols_pred = np.array(cols)

    TP_gt, FN = 0, 0
    for i in range(M):
        if not i in rows_gt:
            FN += 1
            continue
        iou_i = iou_data_gt[rows_gt == i]
        col_i = cols_gt[rows_gt == i]
        argmax_iou = np.argmax(iou_i)
        max_iou = iou_i[argmax_iou]
        if max_iou > 0.1:
            TP_gt += 1
            iou_data_gt[cols_gt==col_i[argmax_iou]] = 0
        else:
            FN += 1

    TP_pred, FP = 0, 0
    for j in range(N):
        if not j in cols_pred:
            FP += 1
            continue
        iou_j = iou_data_pred[cols_pred == j]
        row_j = rows_pred[cols_pred == j]
        argmax_iou = np.argmax(iou_j)
        max_iou = iou_j[argmax_iou]
        if max_iou > 0.1:
            TP_pred += 1
            iou_data_pred[rows_pred==row_j[argmax_iou]] = 0
        else:
            FP += 1

    return TP_gt, TP_pred, FN, FP


def compute_detection_metrics(mask_ref: np.ndarray, mask_pred: np.ndarray, footprint: int = 0, 
                              spacing: tuple[float] = (1., 1., 1.), ignore_mask: Optional[np.ndarray] = None):
    gt_mask = mask_ref if ignore_mask is None else mask_ref & ~ignore_mask
    pred_mask = mask_pred if ignore_mask is None else mask_pred & ~ignore_mask

    gt_inst, gt_inst_num, pred_inst, pred_inst_num = get_instances(gt_mask, pred_mask, footprint=footprint, spacing=spacing)
    TP_gt, TP_pred, FN, FP = get_inst_TPFPFN(gt_inst, gt_inst_num, pred_inst, pred_inst_num)
    recall = TP_gt / (TP_gt + FN) if TP_gt + FN > 0 else 1
    precision = TP_pred / (TP_pred + FP) if TP_pred + FP > 0 else 1
    F1 = (2 * recall * precision) / (recall + precision) if recall + precision > 0 else 0
    return F1, recall, precision, TP_gt, TP_pred, FP, FN