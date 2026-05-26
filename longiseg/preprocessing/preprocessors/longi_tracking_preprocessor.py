from typing import List, Union

import multiprocessing
import shutil
from time import sleep

import cc3d
import edt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir, isfile, join, load_json, save_json, os_split_path
from tqdm import tqdm

from longiseg.paths import LongiSeg_preprocessed, LongiSeg_raw
from longiseg.preprocessing.resampling.default_resampling import compute_new_shape
from longiseg.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from longiseg.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from longiseg.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from longiseg.utilities.utils import get_filenames_of_train_images_and_targets

from longiseg.preprocessing.preprocessors.longi_preprocessor import LongiSegPreprocessor


class LongiSegTrackingPreprocessor(LongiSegPreprocessor):
    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
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
        else:
            seg = np.zeros((1, *data.shape[1:]), dtype=np.int8)
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        properties['bbox_used_for_cropping'] = [[None, None] for _ in range(len(shape_before_cropping))]
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
            properties['class_locations'] = self._sample_foreground_locations(seg, verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager, dataset_json: Union[dict, str]):
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
                                      dataset_json)
        return data, seg, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
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

        return properties

    def run_patient(self, patient: str, patient_scans: list, dataset: dict, plans_manager: PlansManager,
                    configuration_manager: ConfigurationManager, dataset_json: Union[dict, str],
                    output_directory: str):
        tracking_raw = self.tracking[patient]
        patient_spacings = {}
        for s in patient_scans:
            output_filename_truncated = join(output_directory, s)
            image_files = dataset[s]["images"]
            seg_file = dataset[s]["label"]
            properties = self.run_case_save(output_filename_truncated, image_files, seg_file, plans_manager, configuration_manager,
                               dataset_json)
            original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
            patient_spacings[s] = original_spacing
        target_spacing = configuration_manager.spacing
        if isinstance(tracking_raw, dict):
            tracking_preprocessed = {}
            for l in tracking_raw.keys():
                bl_point = tracking_raw[l]['bl_point']
                fu_point_prop = tracking_raw[l]['fu_point_prop']
                fu_point = tracking_raw[l]['fu_point']
                img_bl = tracking_raw[l]['img_bl']
                img_fu = tracking_raw[l]['img_fu']
                bl_spacing = patient_spacings[img_bl]
                fu_spacing = patient_spacings[img_fu]
                # bl_point and fu_point are given as coordinates in the original image space.
                # We need to convert them to the new space.
                if not np.isnan(bl_point).any():
                    bl_point = bl_point[::-1]
                    bl_point = [bl_point[i] for i in plans_manager.transpose_forward]
                    bl_point = [b * bl_spacing[i] / target_spacing[i]
                                for i, b in enumerate(bl_point)]
                if not np.isnan(fu_point_prop).any():
                    fu_point_prop = fu_point_prop[::-1]
                    fu_point_prop = [fu_point_prop[i] for i in plans_manager.transpose_forward]
                    fu_point_prop = [f * fu_spacing[i] / target_spacing[i]
                                    for i, f in enumerate(fu_point_prop)]
                if not np.isnan(fu_point).any():
                    fu_point = fu_point[::-1]
                    fu_point = [fu_point[i] for i in plans_manager.transpose_forward]
                    fu_point = [f * fu_spacing[i] / target_spacing[i]
                                for i, f in enumerate(fu_point)]
                tracking_preprocessed[l] = {
                    'bl_point': bl_point,
                    'fu_point_prop': fu_point_prop,
                    'fu_point': fu_point,
                    'img_bl': tracking_raw[l]['img_bl'],
                    'img_fu': tracking_raw[l]['img_fu'],
                    'merged_lesions': tracking_raw[l]['merged_lesions'],
                }
        elif isinstance(tracking_raw, list|tuple):
            tracking_preprocessed = []
            for scan_dict in tracking_raw:
                scan_dict_preprocessed = {}
                for l in scan_dict.keys():
                    bl_point = scan_dict[l]['bl_point']
                    fu_point_prop = scan_dict[l]['fu_point_prop']
                    fu_point = scan_dict[l]['fu_point']
                    img_bl = scan_dict[l]['img_bl']
                    img_fu = scan_dict[l]['img_fu']
                    bl_spacing = patient_spacings[img_bl]
                    fu_spacing = patient_spacings[img_fu]
                    # bl_point and fu_point are given as coordinates in the original image space.
                    # We need to convert them to the new space.
                    if not np.isnan(bl_point).any():
                        bl_point = bl_point[::-1]
                        bl_point = [bl_point[i] for i in plans_manager.transpose_forward]
                        bl_point = [b * bl_spacing[i] / target_spacing[i]
                                    for i, b in enumerate(bl_point)]
                    if not np.isnan(fu_point_prop).any():
                        fu_point_prop = fu_point_prop[::-1]
                        fu_point_prop = [fu_point_prop[i] for i in plans_manager.transpose_forward]
                        fu_point_prop = [f * fu_spacing[i] / target_spacing[i]
                                        for i, f in enumerate(fu_point_prop)]
                    if not np.isnan(fu_point).any():
                        fu_point = fu_point[::-1]
                        fu_point = [fu_point[i] for i in plans_manager.transpose_forward]
                        fu_point = [f * fu_spacing[i] / target_spacing[i]
                                    for i, f in enumerate(fu_point)]
                    scan_dict_preprocessed[l] = {
                        'bl_point': bl_point,
                        'fu_point_prop': fu_point_prop,
                        'fu_point': fu_point,
                        'img_bl': scan_dict[l]['img_bl'],
                        'img_fu': scan_dict[l]['img_fu'],
                        'merged_lesions': scan_dict[l]['merged_lesions'],
                    }
                tracking_preprocessed.append(scan_dict_preprocessed)
        else:
            raise ValueError(f"Unexpected format for tracking data: {type(tracking_raw)}")
        save_json(tracking_preprocessed, join(output_directory, f'{patient}.json'), sort_keys=False)
        save_json(tracking_preprocessed, join(os_split_path(output_directory)[0], "gt_segmentations", f'{patient}.json'),
                  sort_keys=False)

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

        dataset = get_filenames_of_train_images_and_targets(join(LongiSeg_raw, dataset_name), dataset_json)

        patients = load_json(join(output_directory, "patientsTr.json"))
        self.tracking = load_json(join(LongiSeg_raw, dataset_name, "trackingTr.json"))

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
                                           dataset_json, output_directory),)))

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

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, seed: int = 1234, verbose: bool = False):
        rndst = np.random.RandomState(seed)
        num_samples = 1000
        if np.max(seg) > 127:
            stats = cc3d.statistics(seg[0].astype(np.uint16))
        else:
            stats = cc3d.statistics(seg[0].astype(np.uint8))
        bboxs = stats["bounding_boxes"]
        voxel_counts = stats["voxel_counts"]
        class_locs = {}
        for i, bbox in enumerate(bboxs[1:], 1):
            if voxel_counts[i] == 0:
                continue
            dil_bbox = tuple(slice(max(0, b.start - 1), min(seg.shape[i+1], b.stop + 1)) for i, b in enumerate(bbox))
            seg_slice = seg[(0, *dil_bbox)] == i
            seg_edt = edt.edt(seg_slice)
            coords = np.argwhere(seg_slice)
            edt_values = seg_edt[seg_slice]
            sample_size = min(num_samples, len(coords))
            indices = rndst.choice(len(coords), sample_size, replace=False)
            selected_coords = coords[indices]
            selected_edt_values = edt_values[indices]
            shifted_coords = selected_coords + np.array([b.start for b in dil_bbox])
            # add trailing zero to all shifted coordinates to make them 4D
            full_coords = np.concatenate((np.zeros((len(shifted_coords), 1)), shifted_coords), axis=1)
            class_locs[i] = {
                'coords': full_coords,
                'edt_values': selected_edt_values
            }
        return class_locs


class LongiSegTrackingPretrainingPreprocessor(LongiSegTrackingPreprocessor):
    def run_patient(self, patient: str, patient_scans: list, dataset: dict, plans_manager: PlansManager,
                    configuration_manager: ConfigurationManager, dataset_json: Union[dict, str],
                    output_directory: str):
        lesions = None
        for s in patient_scans:
            output_filename_truncated = join(output_directory, s)
            image_files = dataset[s]["images"]
            seg_file = dataset[s]["label"]
            props = self.run_case_save(output_filename_truncated, image_files, seg_file,
                                                        plans_manager, configuration_manager, dataset_json)
            if lesions is None:
                lesions = set(props['class_locations'].keys())
            else:
                lesions = lesions.intersection(set(props['class_locations'].keys()))
        return bool(lesions)

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

        dataset = get_filenames_of_train_images_and_targets(join(LongiSeg_raw, dataset_name), dataset_json)

        patients = load_json(join(LongiSeg_raw, dataset_name, "patientsTr.json"))

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
                                           dataset_json, output_directory),)))

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

        patients_filtered = dict()
        for patient, i in zip(patients.keys(), r):
            if i.get()[0]:
                patients_filtered[patient] = patients[patient]
            else:
                print(f"Deleting patient {patient} because it has no lesions with coordinates in both follow-up and baseline data.")

        save_json(patients_filtered, join(output_directory, "patientsTr.json"), sort_keys=False)