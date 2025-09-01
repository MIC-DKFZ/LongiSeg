import multiprocessing
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json
from longiseg.configuration import default_num_processes
from longiseg.imageio.base_reader_writer import BaseReaderWriter
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from longiseg.utilities.json_export import recursive_fix_for_json_export
from longiseg.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from longiseg.utilities.plans_handling.plans_handler import PlansManager

from longiseg.evaluation.metrics.autopet_metrics import get_fp_volume, get_fn_volume, compute_volumetric_metrics


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


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


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


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


def compute_metrics(reference_file: str, prediction_files: List[str], tracking_file: str, image_reader_writer: BaseReaderWriter,
                    file_ending: str, ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    spacing = seg_ref_dict['spacing']
    distance_threshold = 1

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_files'] = prediction_files
    results['metrics'] = {}

    tracking_json = load_json(tracking_file)

    skip = []
    for lesion in tracking_json.keys():
        if not isinstance(tracking_json[lesion]["fu_point_prop"], list):
            continue
        merged = tracking_json[lesion]["merged_lesions"]
        lesion = int(lesion)
        if lesion in skip:
            continue
        skip.extend(merged)
        if merged[0] == 0:
            pred_files = [f for f in prediction_files if f.endswith(f"_{lesion}{file_ending}")]
        else:
            pred_files = [f for f in prediction_files if any([f.endswith(f"_{m}{file_ending}") for m in merged])]
        if len(pred_files) == 0:
            # lesion is not in the current scan
            continue
        seg_pred = None
        for pf in pred_files:
            if seg_pred is None:
                seg_pred, _ = image_reader_writer.read_seg(pf)
            else:
                seg_pred_here, _ = image_reader_writer.read_seg(pf)
                seg_pred = np.maximum(seg_pred, seg_pred_here)
        if merged[0] == 0:
            mask_ref = np.zeros_like(seg_ref[0], dtype=np.bool)
        else:
            mask_ref = np.where(np.isin(seg_ref[0], merged), True, False)
        mask_pred = seg_pred[0].astype(np.bool)
        dice, recall, precision = compute_volumetric_metrics(mask_ref, mask_pred, ignore_mask)
        fn_volume = get_fn_volume(mask_ref, mask_pred)
        fp_volume = get_fp_volume(mask_ref, mask_pred)
        results['metrics'][lesion] = {}
        results['metrics'][lesion]['Dice'] = dice
        results['metrics'][lesion]['Recall'] = recall
        results['metrics'][lesion]['Precision'] = precision
        results['metrics'][lesion]['FN_volume'] = fn_volume
        results['metrics'][lesion]['FP_volume'] = fp_volume

    results["metrics"]["mean"] = {
        m: np.nanmean([results['metrics'][r][m] for r in results['metrics'].keys() if r != 'mean'])
        for m in results['metrics'][list(results['metrics'].keys())[0]].keys()
    }
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
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
    files_ref_all = subfiles(folder_ref, suffix=file_ending, join=False)
    files_pred_all = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = []
    files_pred = []
    tracking_files = []
    for ref_file in files_ref_all:
        case_name = ref_file[:-len(file_ending)]
        pred_cases = sorted([j[:-len(file_ending)] for j in files_pred_all if j.startswith(case_name)], key=lambda x: int(x.split("_")[-1]))
        if pred_cases:
            files_ref.append(join(folder_ref, ref_file))
            files_pred.append([join(folder_pred, j + file_ending) for j in pred_cases])
            tracking_files.append(join(folder_ref, case_name.split("_FU_")[0] + ".json"))

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, tracking_files, [image_reader_writer] * len(files_pred),
                     [file_ending] * len(files_pred), [ignore_label] * len(files_pred)))
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
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, 
                               plans_file: str,
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

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/longi_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile,
                                     args.o, args.np, chill=args.chill)