#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Union

import os
import numpy as np

from longiseg.paths import LongiSeg_preprocessed, LongiSeg_raw, LongiSeg_results
from batchgenerators.utilities.file_and_folder_operations import subdirs, isdir


def find_candidate_datasets(dataset_id: int):
    startswith = "Dataset%03.0d" % dataset_id
    if LongiSeg_preprocessed is not None and isdir(LongiSeg_preprocessed):
        candidates_preprocessed = subdirs(LongiSeg_preprocessed, prefix=startswith, join=False)
    else:
        candidates_preprocessed = []

    if LongiSeg_raw is not None and isdir(LongiSeg_raw):
        candidates_raw = subdirs(LongiSeg_raw, prefix=startswith, join=False)
    else:
        candidates_raw = []

    candidates_trained_models = []
    if LongiSeg_results is not None and isdir(LongiSeg_results):
        candidates_trained_models += subdirs(LongiSeg_results, prefix=startswith, join=False)

    all_candidates = candidates_preprocessed + candidates_raw + candidates_trained_models
    unique_candidates = np.unique(all_candidates)
    return unique_candidates


def convert_id_to_dataset_name(dataset_id: int):
    unique_candidates = find_candidate_datasets(dataset_id)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. Please correct that. (I looked in the "
                           "following folders:\n%s\n%s\n%s" % (dataset_id, LongiSeg_raw, LongiSeg_preprocessed, LongiSeg_results))
    if len(unique_candidates) == 0:
        raise RuntimeError(f"Could not find a dataset with the ID {dataset_id}. Make sure the requested dataset ID "
                           f"exists and that nnU-Net knows where raw and preprocessed data are located "
                           f"(see Documentation - Installation). Here are your currently defined folders:\n"
                           f"LongiSeg_preprocessed={os.environ.get('LongiSeg_preprocessed') if os.environ.get('LongiSeg_preprocessed') is not None else 'None'}\n"
                           f"LongiSeg_results={os.environ.get('LongiSeg_results') if os.environ.get('LongiSeg_results') is not None else 'None'}\n"
                           f"LongiSeg_raw={os.environ.get('LongiSeg_raw') if os.environ.get('LongiSeg_raw') is not None else 'None'}\n"
                           f"If something is not right, adapt your environment variables.")
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:10])
    return dataset_id


def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct dataset name. Your input: %s" % dataset_name_or_id)
    return convert_id_to_dataset_name(dataset_name_or_id)
