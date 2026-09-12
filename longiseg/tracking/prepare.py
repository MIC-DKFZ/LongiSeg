import argparse
import os
from typing import Dict, List, Sequence, Tuple

import cc3d
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json

from longiseg.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from longiseg.tracking.tracking_file import (NAN, LesionEntry, TrackingFile, is_missing, is_pair_spec,
                                                       label_file, scan_pair_of)
from longiseg.tracking.registration import PointPropagator


def lesion_stats(seg: np.ndarray) -> Dict[int, Tuple[np.ndarray, int]]:
    seg = seg.astype(np.uint16) if seg.max() > 255 else seg.astype(np.uint8)
    stats = cc3d.statistics(seg)
    counts, centroids = stats["voxel_counts"], stats["centroids"]
    return {label: (np.asarray(centroids[label][::-1], dtype=float), int(counts[label]))
            for label in range(1, len(counts)) if counts[label] > 0}


def centroid(stats: Dict[int, Tuple[np.ndarray, int]], labels: Sequence[int]) -> List[float]:
    # the centroid of several lesions is the mean of their centroids, weighted by how many voxels each contributes
    total = sum(stats[label][1] for label in labels)
    center = sum(stats[label][0] * stats[label][1] for label in labels) / total
    return [float(c) for c in center]


class SegmentationLesions:
    def __init__(self, labels_folder: str, reader_writer, file_ending: str):
        self.labels_folder = labels_folder
        self.reader_writer = reader_writer
        self.file_ending = file_ending
        self._cache: Dict[str, Dict[int, Tuple[np.ndarray, int]]] = {}

    def __call__(self, identifier: str) -> Dict[int, Tuple[np.ndarray, int]]:
        if identifier not in self._cache:
            seg_file = label_file(self.labels_folder, identifier, self.file_ending)
            if not isfile(seg_file):
                raise FileNotFoundError(f"Expected segmentation {seg_file} but it does not exist.")
            seg, _ = self.reader_writer.read_seg(seg_file)
            self._cache[identifier] = lesion_stats(seg[0])
        return self._cache[identifier]


def fill_from_segmentations(tracking: TrackingFile, lesions: SegmentationLesions, overwrite: bool = False) -> None:
    for patient, pair_index, scan_dict in list(tracking.scan_dicts()):
        img_bl, img_fu = scan_pair_of(patient, scan_dict)
        bl_lesions, fu_lesions = lesions(img_bl), lesions(img_fu)

        if is_pair_spec(scan_dict):
            scan_dict = {str(lesion): {"img_bl": img_bl, "img_fu": img_fu} for lesion in sorted(bl_lesions)}
            tracking.set_scan_dict(patient, pair_index, scan_dict)

        for lesion, info in scan_dict.items():
            info.setdefault("img_bl", img_bl)
            info.setdefault("img_fu", img_fu)
            entry = LesionEntry(patient, pair_index, lesion, info)
            if entry.label not in bl_lesions:
                raise RuntimeError(f"Lesion {lesion} of patient {patient} is not present in the segmentation of "
                                   f"{img_bl}.")
            if overwrite or not entry.has("bl_point"):
                entry.bl_point = centroid(bl_lesions, [entry.label])
            if entry.merged_lesions is None:
                entry.merged_lesions = [entry.label] if entry.label in fu_lesions else [0]
            else:
                unknown = [m for m in entry.merged_lesions if m != 0 and m not in fu_lesions]
                if len(unknown) > 0:
                    print(f"[WARN] patient {patient}, lesion {lesion}: merged_lesions {unknown} are not in the "
                          f"segmentation of {img_fu} and are ignored.")
            if overwrite or not entry.has("fu_point"):
                merged = [m for m in entry.merged_lesions if m in fu_lesions]
                entry.fu_point = centroid(fu_lesions, merged) if merged else NAN
            info.setdefault("fu_point_prop", NAN)


def fill_proposals(tracking: TrackingFile, images_folder: str, propagator: PointPropagator, file_ending: str,
                   channel: int = 0, overwrite: bool = False, save_after_each_pair: bool = True) -> None:
    for patient, _, scan_dict in tracking.scan_dicts():
        if is_pair_spec(scan_dict):
            raise RuntimeError(f"Patient {patient} only specifies a scan pair. Run with -labels to expand it into its "
                               f"lesions first, or provide the lesions and their bl_point yourself.")

    todo: Dict[tuple, List[LesionEntry]] = {}
    for entry in tracking:
        if not overwrite and entry.has("fu_point_prop"):
            continue
        if not entry.has("bl_point"):
            print(f"[WARN] patient {entry.patient}, lesion {entry.lesion} has no bl_point, skipping.")
            continue
        todo.setdefault(entry.scan_pair, []).append(entry)

    print(f"Propagating points for {sum(len(v) for v in todo.values())} lesions in {len(todo)} scan pairs")
    for n, ((img_bl, img_fu), entries) in enumerate(todo.items(), 1):
        bl_file = entries[0].bl_image_file(images_folder, file_ending, channel)
        fu_file = entries[0].fu_image_file(images_folder, file_ending, channel)
        if not isfile(bl_file) or not isfile(fu_file):
            print(f"[WARN] missing image for pair {img_bl} -> {img_fu}, skipping {len(entries)} lesions.")
            continue
        print(f"[{n}/{len(todo)}] registering {img_bl} -> {img_fu}")
        try:
            registration = propagator.register(bl_file, fu_file)
        except Exception as e:
            print(f"[WARN] registration of {img_bl} -> {img_fu} failed, skipping {len(entries)} lesions: {e}")
            continue
        for entry in entries:
            try:
                entry.fu_point_prop = registration.propagate(entry.bl_point)
            except Exception as e:
                print(f"[WARN] propagation failed for patient {entry.patient}, lesion {entry.lesion}: {e}")
                entry.info.setdefault("fu_point_prop", NAN)
        del registration
        if save_after_each_pair:
            tracking.save()


def prepare_tracking(tracking_file: str, output_file: str, dataset_json_file: str, labels_folder: str = None,
                     images_folder: str = None, propagator: PointPropagator = None, channel: int = 0,
                     overwrite: bool = False) -> TrackingFile:
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']

    if overwrite or not isfile(output_file):
        tracking = TrackingFile(load_json(tracking_file), output_file)
    else:
        tracking = TrackingFile.load(output_file)

    if labels_folder is not None:
        example = label_file(labels_folder, sorted(os.listdir(labels_folder))[0].split(file_ending)[0], file_ending)
        reader_writer = determine_reader_writer_from_dataset_json(dataset_json, example)()
        fill_from_segmentations(tracking, SegmentationLesions(labels_folder, reader_writer, file_ending), overwrite)
        tracking.save()

    if images_folder is not None:
        if propagator is None:
            raise ValueError("A propagator is required to fill fu_point_prop.")
        fill_proposals(tracking, images_folder, propagator, file_ending, channel, overwrite)

    tracking.save()
    return tracking


def prepare_tracking_entry_point():
    parser = argparse.ArgumentParser(
        description="Fill in the points of a tracking json. Baseline and follow-up points are taken from the "
                    "centroids of the reference segmentations (-labels), the propagated point is obtained by "
                    "registering the two scans (-images). Existing values are kept, so the two steps can be run "
                    "separately and an interrupted run can simply be repeated.")
    parser.add_argument('tracking_file', type=str,
                        help='tracking json to start from. Needs to provide at least img_bl and img_fu per scan pair')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='output file. Optional. Default: tracking_file with _prepared appended')
    parser.add_argument('-djfile', type=str, required=True, help='dataset.json file')
    parser.add_argument('-labels', type=str, required=False, default=None,
                        help='folder with the reference segmentations. If given, lesions are read from the baseline '
                             'segmentation and bl_point, fu_point and merged_lesions are filled in')
    parser.add_argument('-images', type=str, required=False, default=None,
                        help='folder with the images. If given, fu_point_prop is filled in by registering the '
                             'baseline to the follow-up scan')
    parser.add_argument('-model', type=str, default='unigradicon', choices=['unigradicon', 'multigradicon'],
                        help='registration model. Optional. Default: unigradicon')
    parser.add_argument('--fast', action='store_true',
                        help='skip the instance optimization during the registration. Considerably faster, but the '
                             'propagated points are less accurate')
    args = parser.parse_args()

    if args.labels is None and args.images is None:
        parser.error("nothing to do, give -labels, -images or both")

    output_file = args.o
    if output_file is None:
        base, _, ending = args.tracking_file.partition('.json')
        output_file = base + '_prepared.json' + ending

    propagator = None
    if args.images is not None:
        from longiseg.tracking.registration import UniGradIconPropagator
        propagator = UniGradIconPropagator(model=args.model, io_iterations=0 if args.fast else 50)

    tracking = prepare_tracking(args.tracking_file, output_file, args.djfile, args.labels, args.images, propagator)

    counts = tracking.counts()
    print(f"\n{counts['lesions']} tracked lesions")
    for key in ('bl_point', 'fu_point_prop', 'fu_point', 'disappeared', 'needing_verification'):
        print(f"  {key + ':':22s} {counts[key]}")
    print(f"\nwrote {output_file}")


if __name__ == '__main__':
    prepare_tracking_entry_point()
