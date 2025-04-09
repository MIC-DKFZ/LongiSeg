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
from typing import List, Union

import math
import multiprocessing
import shutil
from time import sleep
from typing import Tuple

import SimpleITK
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir, isfile, join, load_json
from tqdm import tqdm

import longiseg
from longiseg.paths import LongiSeg_preprocessed, LongiSeg_raw
from longiseg.preprocessing.cropping.cropping import crop_to_nonzero
from longiseg.preprocessing.resampling.default_resampling import compute_new_shape
from longiseg.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from longiseg.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from longiseg.utilities.find_class_by_name import recursive_find_python_class
from longiseg.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from longiseg.utilities.utils import get_filenames_of_train_images_and_targets

from longiseg.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor


class LongiSegPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str], bbox: list = None):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg, bbox)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager, dataset_json: Union[dict, str],
                 bbox: list = None):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if self.verbose:
            print(seg_file)
        data, seg, data_properties = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json, bbox)
        return data, seg, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str], bbox: list = None):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json, bbox)
        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16, copy=False)
        block_size_data, chunk_size_data = nnUNetDatasetBlosc2.comp_blosc2_params(
            data.shape,
            tuple(configuration_manager.patch_size),
            data.itemsize)
        block_size_seg, chunk_size_seg = nnUNetDatasetBlosc2.comp_blosc2_params(
            seg.shape,
            tuple(configuration_manager.patch_size),
            seg.itemsize)

        nnUNetDatasetBlosc2.save_case(data, seg, properties, output_filename_truncated,
                                      chunks=chunk_size_data, blocks=block_size_data,
                                      chunks_seg=chunk_size_seg, blocks_seg=block_size_seg)

    def run_patient(self, patient: str, patient_scans: list, dataset: dict, plans_manager: PlansManager,
                    configuration_manager: ConfigurationManager, dataset_json: Union[dict, str],
                    output_directory: str, patient_meta_dir: str):
        patient_bbox = load_json(join(patient_meta_dir, f'{patient}.json'))
        for s in patient_scans:
            output_filename_truncated = join(output_directory, s)
            image_files = dataset[s]["images"]
            seg_file = dataset[s]["label"]
            self.run_case_save(output_filename_truncated, image_files, seg_file, plans_manager, configuration_manager,
                               dataset_json, patient_bbox)

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        foreground_mask = seg != 0
        foreground_coords = np.argwhere(foreground_mask)
        seg = seg[foreground_mask]
        del foreground_mask
        unique_labels = pd.unique(seg.ravel())

        # We don't need more than 1e7 foreground samples. That's insanity. Cap here
        if len(foreground_coords) > 1e7:
            take_every = math.floor(len(foreground_coords) / 1e7)
            # keep computation time reasonable
            if verbose:
                print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
            foreground_coords = foreground_coords[::take_every]
            seg = seg[::take_every]

        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)

            # check if any of the labels are in seg, if not skip c
            if isinstance(c, (tuple, list)):
                if not any([ci in unique_labels for ci in c]):
                    class_locs[k] = []
                    continue
            else:
                if c not in unique_labels:
                    class_locs[k] = []
                    continue

            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = foreground_coords[mask]
            else:
                mask = seg == c
                all_locs = foreground_coords[mask]
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
            seg = seg[~mask]
            foreground_coords = foreground_coords[~mask]
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(longiseg.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'longiseg.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(LongiSeg_raw, dataset_name)), "The requested dataset could not be found in LongiSeg_raw"

        plans_file = join(LongiSeg_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(LongiSeg_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(LongiSeg_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        shutil.copy(join(LongiSeg_raw, dataset_name, "patientsTr.json"), join(output_directory, "patientsTr.json"))

        patient_meta_dir = join(LongiSeg_preprocessed, dataset_name, "patient_meta")

        dataset = get_filenames_of_train_images_and_targets(join(LongiSeg_raw, dataset_name), dataset_json)

        patients = load_json(join(output_directory, "patientsTr.json"))

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(patients.keys())))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            for patient, patient_scans in patients.items():
                r.append(p.starmap_async(self.run_patient,
                                         ((patient, patient_scans, dataset, plans_manager, configuration_manager,
                                           dataset_json, output_directory, patient_meta_dir),)))

            with tqdm(desc=None, total=len(patients.keys()), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg