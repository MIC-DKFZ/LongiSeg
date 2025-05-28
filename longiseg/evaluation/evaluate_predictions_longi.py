import multiprocessing
from copy import deepcopy
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from typing import Tuple, List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from longiseg.configuration import default_num_processes
from longiseg.imageio.base_reader_writer import BaseReaderWriter
from longiseg.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from longiseg.utilities.json_export import recursive_fix_for_json_export
from longiseg.utilities.plans_handling.plans_handler import PlansManager

from longiseg.evaluation.metrics.volumetric_metrics import compute_volumetric_metrics
from longiseg.evaluation.metrics.distance_metrics import SSD
from longiseg.evaluation.metrics.detection_metrics import compute_detection_metrics
from longiseg.evaluation.metrics.surface_metrics import NSD


def fix_and_save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    but we do it better than nnUNet
    """
    if not isinstance(results, dict):
        return results
    results_converted = dict()
    for k, v in results.items():
        if isinstance(k, Tuple):
            k = str(k)
        results_converted[k] = fix_and_save_summary_json(v, None)
    if output_file is not None:
        save_json(results_converted, output_file, sort_keys=True)
    else:
        return results_converted


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_longi_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None, distance_threshold: int = 1,
                    size_threshold: int = 0, footprint: int = 0) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    spacing = seg_ref_dict['spacing']
    voxel_vol = np.prod(spacing)
    size_threshold_vox = np.ceil(size_threshold / voxel_vol).astype(int)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref[0], r)
        mask_pred = region_or_label_to_mask(seg_pred[0], r)
        mask_ref = remove_small_objects(mask_ref, min_size=size_threshold_vox).astype(mask_ref.dtype)
        mask_pred = remove_small_objects(mask_pred, min_size=size_threshold_vox).astype(mask_pred.dtype)
        dice, iou, recall, precision, tp, fp, fn, tn = compute_volumetric_metrics(mask_ref, mask_pred, ignore_mask)
        nsd = NSD(mask_ref, mask_pred, spacing, distance_threshold, ignore_mask)
        assd, hd95 = SSD(mask_ref, mask_pred, spacing, ignore_mask)
        F1, inst_recall, inst_precision, inst_TP_gt, inst_TP_pred, inst_FP, inst_FN = \
            compute_detection_metrics(mask_ref, mask_pred, footprint, spacing, ignore_mask)
        results['metrics'][r]['Dice'] = dice
        results['metrics'][r]['IoU'] = iou
        results['metrics'][r]['Recall'] = recall
        results['metrics'][r]['Precision'] = precision
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
        results['metrics'][r]['NSD'] = nsd
        results['metrics'][r]['ASSD'] = assd
        results['metrics'][r]['HD95'] = hd95
        results['metrics'][r]['F1'] = F1
        results['metrics'][r]['Instance_Recall'] = inst_recall
        results['metrics'][r]['Instance_Precision'] = inst_precision
        results['metrics'][r]['Instance_TP_gt'] = inst_TP_gt
        results['metrics'][r]['Instance_TP_pred'] = inst_TP_pred
        results['metrics'][r]['Instance_FP'] = inst_FP
        results['metrics'][r]['Instance_FN'] = inst_FN
        results['metrics'][r]['Instance_n_pred'] = inst_FP + inst_TP_pred
        results['metrics'][r]['Instance_n_ref'] = inst_FN + inst_TP_gt
    return results


def compute_longi_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              patients_json: dict|None,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)

    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    # filter files with patients
    if patients_json is not None:
        patients_filtered = {k: [i for i in v if i+file_ending in files_pred] for k, v in patients_json.items()}
        patients_filtered = {k: v for k, v in patients_filtered.items() if len(v) > 0}
        files_ref = [join(folder_ref, i + file_ending) for v in patients_json.values() for i in v if i+file_ending in files_pred]
        files_pred = [join(folder_pred, i + file_ending) for v in patients_json.values() for i in v if i+file_ending in files_pred]
    else:
        files_ref = [join(folder_ref, i) for i in files_pred]
        files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_longi_metrics(*i)
        results = pool.starmap(
            compute_longi_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())

    if patients_json is not None:
        patient_results = {}
        index = 0
        for k, v in patients_filtered.items():
            patient_results[k] = {"per_case": {}}
            for i in v:
                patient_results[k]["per_case"][i] = results[index]
                index += 1

            # mean per patient
            patient_means = {}
            for r in regions_or_labels:
                patient_means[r] = {}
                for m in metric_list:
                    patient_means[r][m] = np.nanmean([i['metrics'][r][m] for i in patient_results[k]["per_case"].values()])
            patient_results[k]['mean'] = patient_means

        # mean over all patients
        means = {}
        for r in regions_or_labels:
            means[r] = {}
            for m in metric_list:
                means[r][m] = np.nanmean([v["mean"][r][m] for v in patient_results.values()])

        # foreground mean over all patients
        foreground_means = {}
        for m in metric_list:
            values = []
            for k in means.keys():
                if k == 0 or k == '0':
                    continue
                values.append(means[k][m])
            foreground_means[m] = np.mean(values)
        recursive_fix_for_json_export(patient_results)
        recursive_fix_for_json_export(means)
        recursive_fix_for_json_export(foreground_means)
        result = {'metric_per_patient': patient_results, 'mean': means, 'foreground_mean': foreground_means}

    else:
        means = {}
        for r in regions_or_labels:
            means[r] = {}
            for m in metric_list:
                means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

        # foreground mean
        foreground_mean = {}
        for m in metric_list:
            values = []
            for k in means.keys():
                if k == 0 or k == '0':
                    continue
                values.append(means[k][m])
            foreground_mean[m] = np.mean(values)
        [recursive_fix_for_json_export(i) for i in results]
        recursive_fix_for_json_export(means)
        recursive_fix_for_json_export(foreground_mean)
        result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}

    if output_file is not None:
        fix_and_save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_longi_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, 
                               plans_file: str,
                               patients_file: str = None,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'longi_summary.json')

    if patients_file is None:
        patients_json = None
    else:
        patients_json = load_json(patients_file)

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_longi_metrics_on_folder(folder_ref, folder_pred, output_file, patients_json, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def evaluate_longi_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-patfile', type=str, required=False,
                        help='patients.json file, if None, evaluation will not be split up by patients')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/longi_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_longi_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile,
                                     args.patfile, args.o, args.np, chill=args.chill)