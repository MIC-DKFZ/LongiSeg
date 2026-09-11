import multiprocessing
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json
from longiseg.configuration import default_num_processes
from longiseg.imageio.base_reader_writer import BaseReaderWriter
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from longiseg.utilities.json_export import recursive_fix_for_json_export
from longiseg.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from longiseg.utilities.plans_handling.plans_handler import PlansManager

from longiseg.evaluation.metrics.tracking_metrics import get_fp_volume, get_fn_volume, compute_volumetric_metrics


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def scan_dicts_from_patient_tracking(tracking: Union[dict, list], source: str) -> List[dict]:
    scan_dicts = tracking if isinstance(tracking, list) else [tracking]
    for scan_dict in scan_dicts:
        for lesion, info in scan_dict.items():
            if not isinstance(info, dict) or "img_fu" not in info:
                raise RuntimeError(f"{source} is not a valid tracking file: entry {lesion} has no 'img_fu'.")
    return scan_dicts


def scan_dicts_from_dataset_tracking(tracking: dict, source: str) -> List[dict]:
    return [scan_dict for patient, patient_tracking in tracking.items()
            for scan_dict in scan_dicts_from_patient_tracking(patient_tracking, f"{source} (patient {patient})")]


def index_lesions_by_follow_up(scan_dicts: List[dict]) -> Dict[str, dict]:
    lesions_per_scan = {}
    for scan_dict in scan_dicts:
        for lesion, info in scan_dict.items():
            lesions_per_scan.setdefault(info["img_fu"], {})[lesion] = info
    return lesions_per_scan


def load_lesions_per_scan(folder_ref: str, tracking_file: str = None) -> Dict[str, dict]:
    if tracking_file is not None:
        return index_lesions_by_follow_up(scan_dicts_from_dataset_tracking(load_json(tracking_file), tracking_file))
    tracking_files = subfiles(folder_ref, suffix='.json', join=True)
    if len(tracking_files) == 0:
        raise RuntimeError(f"Did not find any tracking json in {folder_ref}. When evaluating predictions that were "
                           f"not made during training, pass the tracking file of the dataset explicitly.")
    return index_lesions_by_follow_up([scan_dict for f in tracking_files
                                       for scan_dict in scan_dicts_from_patient_tracking(load_json(f), f)])


def find_prediction_files(folder_pred: str, predictions: set, case_name: str, lesions: Iterable,
                          file_ending: str) -> Dict[int, str]:
    pred_files = {}
    for lesion in lesions:
        for name in (f"{case_name}_{lesion}{file_ending}", f"{case_name}_lesion_{lesion}{file_ending}"):
            if name in predictions:
                pred_files[int(lesion)] = join(folder_pred, name)
                break
    return pred_files


def compute_metrics(reference_file: str, pred_files: Dict[int, str], tracked_lesions: dict,
                    image_reader_writer: BaseReaderWriter, ignore_label: int = None) -> dict:
    # load images
    seg_ref, _ = image_reader_writer.read_seg(reference_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_files'] = sorted(pred_files.values())
    results['metrics'] = {}

    skip = []
    for lesion, info in tracked_lesions.items():
        if not isinstance(info["fu_point_prop"], list):
            continue
        merged = info["merged_lesions"]
        lesion = int(lesion)
        if lesion in skip:
            continue
        skip.extend(merged)
        if merged[0] == 0:
            merged_here = [lesion]
        else:
            merged_here = merged
        files_here = [pred_files[m] for m in merged_here if m in pred_files]
        if len(files_here) == 0:
            # lesion is not in the current scan
            continue
        seg_pred = None
        for pf in files_here:
            if seg_pred is None:
                seg_pred, _ = image_reader_writer.read_seg(pf)
            else:
                seg_pred_here, _ = image_reader_writer.read_seg(pf)
                seg_pred = np.maximum(seg_pred, seg_pred_here)
        if merged[0] == 0:
            mask_ref = np.zeros_like(seg_ref[0], dtype=np.bool_)
        else:
            mask_ref = np.where(np.isin(seg_ref[0], merged), True, False)
        mask_pred = seg_pred[0].astype(np.bool_)
        dice, recall, precision = compute_volumetric_metrics(mask_ref, mask_pred, ignore_mask)
        results['metrics'][lesion] = {
            'Dice': dice,
            'Recall': recall,
            'Precision': precision,
            'FN_volume': get_fn_volume(mask_ref, mask_pred),
            'FP_volume': get_fp_volume(mask_ref, mask_pred),
        }

    if len(results['metrics']) == 0:
        raise RuntimeError(f"No tracked lesion of {reference_file} could be evaluated. Please check that the "
                           f"predictions and the tracking info belong to this dataset.")

    results["metrics"]["mean"] = {
        m: np.nanmean([results['metrics'][r][m] for r in results['metrics'].keys() if r != 'mean'])
        for m in results['metrics'][list(results['metrics'].keys())[0]].keys()
    }
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              tracking_file: str = None) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'

    lesions_per_scan = load_lesions_per_scan(folder_ref, tracking_file)

    predictions = set(subfiles(folder_pred, suffix=file_ending, join=False))
    files_ref = []
    all_pred_files = []
    all_tracked_lesions = []
    for ref_file in subfiles(folder_ref, suffix=file_ending, join=False):
        case_name = ref_file[:-len(file_ending)]
        tracked_lesions = lesions_per_scan.get(case_name)
        if tracked_lesions is None:
            continue
        pred_files = find_prediction_files(folder_pred, predictions, case_name, tracked_lesions.keys(), file_ending)
        if len(pred_files) == 0:
            continue
        files_ref.append(join(folder_ref, ref_file))
        all_pred_files.append(pred_files)
        all_tracked_lesions.append(tracked_lesions)

    if len(files_ref) == 0:
        raise RuntimeError(f"Did not find any prediction in {folder_pred} matching a reference in {folder_ref}.")

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, all_pred_files, all_tracked_lesions,
                     [image_reader_writer] * len(files_ref), [ignore_label] * len(files_ref)))
        )

    # mean metric per class
    initial_labels = list(results[0]["metrics"].keys())
    metric_list = list(results[0]["metrics"][initial_labels[0]].keys())
    means = {}
    for m in metric_list:
        means[m] = np.nanmean([i['metrics']['mean'][m] for i in results])

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    result = {'metric_per_case': results, 'mean': means}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str,
                               plans_file: str,
                               tracking_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'longi_summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending, lm.ignore_label, num_processes,
                              tracking_file)


def evaluate_tracking_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations, one file per lesion')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/longi_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('-tfile', type=str, required=True,
                        help='tracking json of the dataset, the same file that was passed to LongiSeg_predict_tracking')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.tfile, args.o, args.np)


if __name__ == "__main__":
    evaluate_tracking_folder_entry_point()
