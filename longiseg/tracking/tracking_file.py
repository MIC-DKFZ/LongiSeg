import os
import tempfile
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, save_json

NAN = float('nan')

Point = Union[List[float], float]


def is_missing(point) -> bool:
    if point is None:
        return True
    try:
        return bool(np.isnan(np.asarray(point, dtype=float)).any())
    except (TypeError, ValueError):
        return True


def as_point(point) -> Point:
    if is_missing(point):
        return NAN
    point = [float(c) for c in point]
    if len(point) != 3:
        raise ValueError(f"A point needs 3 coordinates, got {point}.")
    return point


def image_file(images_folder: str, identifier: str, file_ending: str, channel: int = 0) -> str:
    return join(images_folder, f"{identifier}_{channel:04d}{file_ending}")


def label_file(labels_folder: str, identifier: str, file_ending: str) -> str:
    return join(labels_folder, identifier + file_ending)


class LesionEntry:
    """One tracked lesion. Writes through to the underlying tracking dict, so setting a point here and calling
    TrackingFile.save() is all an interactive tool needs to do."""

    def __init__(self, patient: str, pair_index: int, lesion: str, info: dict):
        self.patient = patient
        self.pair_index = pair_index
        self.lesion = lesion
        self.info = info

    @property
    def label(self) -> int:
        return int(self.lesion)

    @property
    def img_bl(self) -> str:
        return self.info["img_bl"]

    @property
    def img_fu(self) -> str:
        return self.info["img_fu"]

    @property
    def scan_pair(self) -> Tuple[str, str]:
        return self.img_bl, self.img_fu

    @property
    def merged_lesions(self) -> Optional[List[int]]:
        return self.info.get("merged_lesions")

    @merged_lesions.setter
    def merged_lesions(self, value: Sequence[int]):
        self.info["merged_lesions"] = [int(v) for v in value]

    @property
    def disappeared(self) -> bool:
        merged = self.merged_lesions
        return merged is not None and len(merged) > 0 and merged[0] == 0

    def _get_point(self, key) -> Point:
        return self.info.get(key, NAN)

    def _set_point(self, key, value):
        self.info[key] = as_point(value)

    @property
    def bl_point(self) -> Point:
        return self._get_point("bl_point")

    @bl_point.setter
    def bl_point(self, value):
        self._set_point("bl_point", value)

    @property
    def fu_point_prop(self) -> Point:
        return self._get_point("fu_point_prop")

    @fu_point_prop.setter
    def fu_point_prop(self, value):
        self._set_point("fu_point_prop", value)

    @property
    def fu_point(self) -> Point:
        return self._get_point("fu_point")

    @fu_point.setter
    def fu_point(self, value):
        self._set_point("fu_point", value)

    def has(self, key: str) -> bool:
        return not is_missing(self.info.get(key))

    def needs_verification(self) -> bool:
        return self.has("fu_point_prop") and not self.has("fu_point")

    def bl_image_file(self, images_folder: str, file_ending: str, channel: int = 0) -> str:
        return image_file(images_folder, self.img_bl, file_ending, channel)

    def fu_image_file(self, images_folder: str, file_ending: str, channel: int = 0) -> str:
        return image_file(images_folder, self.img_fu, file_ending, channel)

    def __repr__(self):
        return (f"LesionEntry({self.patient}, lesion {self.lesion}, {self.img_bl} -> {self.img_fu}, "
                f"bl_point={self.bl_point}, fu_point_prop={self.fu_point_prop}, fu_point={self.fu_point})")


class TrackingFile:
    """The tracking json of a dataset. Patients map either to one {lesion: info} dict or, when a lesion is tracked
    through more than two timepoints, to a list of them, one per consecutive baseline/follow-up pair."""

    def __init__(self, tracking: dict, path: str = None):
        self.tracking = tracking
        self.path = path

    @classmethod
    def load(cls, path: str) -> "TrackingFile":
        return cls(load_json(path), path)

    @classmethod
    def load_or_create(cls, path: str, source: str) -> "TrackingFile":
        return cls(load_json(path), path) if isfile(path) else cls(load_json(source), path)

    def save(self, path: str = None) -> str:
        path = path or self.path
        if path is None:
            raise ValueError("No path to save to. Pass one or load the TrackingFile from a file.")
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=folder, suffix=".tmp") as f:
            tmp = f.name
        save_json(self.tracking, tmp, sort_keys=False)
        os.replace(tmp, path)
        self.path = path
        return path

    def scan_dicts(self) -> Iterator[Tuple[str, int, dict]]:
        for patient, patient_tracking in self.tracking.items():
            if isinstance(patient_tracking, list):
                for i, scan_dict in enumerate(patient_tracking):
                    yield patient, i, scan_dict
            else:
                yield patient, 0, patient_tracking

    def set_scan_dict(self, patient: str, pair_index: int, scan_dict: dict) -> None:
        if isinstance(self.tracking[patient], list):
            self.tracking[patient][pair_index] = scan_dict
        else:
            self.tracking[patient] = scan_dict

    def __iter__(self) -> Iterator[LesionEntry]:
        for patient, pair_index, scan_dict in self.scan_dicts():
            if is_pair_spec(scan_dict):
                continue
            for lesion, info in scan_dict.items():
                yield LesionEntry(patient, pair_index, lesion, info)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def entries(self) -> List[LesionEntry]:
        return list(self)

    def needing_verification(self) -> List[LesionEntry]:
        return [e for e in self if e.needs_verification()]

    def by_scan_pair(self) -> Dict[Tuple[str, str], List[LesionEntry]]:
        pairs: Dict[Tuple[str, str], List[LesionEntry]] = {}
        for entry in self:
            pairs.setdefault(entry.scan_pair, []).append(entry)
        return pairs

    def counts(self) -> Dict[str, int]:
        entries = self.entries()
        return {
            "lesions": len(entries),
            "bl_point": sum(e.has("bl_point") for e in entries),
            "fu_point_prop": sum(e.has("fu_point_prop") for e in entries),
            "fu_point": sum(e.has("fu_point") for e in entries),
            "disappeared": sum(e.disappeared for e in entries),
            "needing_verification": sum(e.needs_verification() for e in entries),
        }


def is_pair_spec(scan_dict: dict) -> bool:
    return "img_bl" in scan_dict and "img_fu" in scan_dict


def scan_pair_of(patient: str, scan_dict: dict) -> Tuple[str, str]:
    if is_pair_spec(scan_dict):
        return scan_dict["img_bl"], scan_dict["img_fu"]
    images = {(info.get("img_bl"), info.get("img_fu")) for info in scan_dict.values()}
    if len(images) != 1 or None in next(iter(images)):
        raise RuntimeError(f"Patient {patient} has an entry whose lesions do not all share one img_bl/img_fu pair: "
                           f"{sorted(images)}.")
    return next(iter(images))
