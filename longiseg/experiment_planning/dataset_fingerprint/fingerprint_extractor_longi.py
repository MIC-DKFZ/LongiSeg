import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from tqdm import tqdm

from longiseg.imageio.base_reader_writer import BaseReaderWriter
from longiseg.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from longiseg.paths import LongiSeg_preprocessed
from longiseg.preprocessing.cropping.cropping import crop_to_nonzero

from longiseg.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor


class DatasetFingerprintExtractorLongiSeg(DatasetFingerprintExtractor):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False):
        super().__init__(dataset_name_or_id, num_processes, verbose)
        self.patients = load_json(join(self.input_folder, 'patientsTr.json'))

    @staticmethod
    def analyze_patient(dataset: dict, patient: str, patient_scans: list, reader_writer_class: Type[BaseReaderWriter], 
                        num_samples: int = 10000, preprocess_output_folder: str = None):
        num_sample_patient = num_samples // len(patient_scans)
        shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
        relative_size_after_cropping, bboxs = [], [], [], [], [], []
        for s in patient_scans:
            image_files = dataset[s]['images']
            segmentation_file = dataset[s]['label']
            case_analysis = DatasetFingerprintExtractorLongiSeg.analyze_case(image_files, segmentation_file,
                                                                        reader_writer_class, num_sample_patient)
            shape_after_crop.append(case_analysis[0])
            spacing.append(case_analysis[1])
            foreground_intensities_per_channel.append(case_analysis[2])
            foreground_intensity_stats_per_channel.append(case_analysis[3])
            relative_size_after_cropping.append(case_analysis[4])
            bboxs.append(case_analysis[5])
        patient_bbox = [[min([b[0] for b in bs]), max((b[1] for b in bs))] for bs in zip(*bboxs)]
        if not preprocess_output_folder is None:
            patient_meta_dir = join(preprocess_output_folder, "patient_meta")
            maybe_mkdir_p(patient_meta_dir)
            save_json(patient_bbox, join(patient_meta_dir, f"{patient}.json"))
        return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
                relative_size_after_cropping

    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter],
                     num_samples: int = 10000):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
            DatasetFingerprintExtractorLongiSeg.collect_foreground_intensities(seg_cropped, data_cropped,
                                                                       num_samples=num_samples)

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
               relative_size_after_cropping, bbox

    def run(self, overwrite_existing: bool = False) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in LongiSeg_preprocessed and LongiSeg_results, so LongiSeg_preprocessed it is
        preprocessed_output_folder = join(LongiSeg_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            # yikes. Rip the following line
                                                                            self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0])

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(self.dataset))

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                # for k in self.dataset.keys():
                #     r.append(p.starmap_async(DatasetFingerprintExtractorLongiSeg.analyze_case,
                #                              ((self.dataset[k]['images'], self.dataset[k]['label'], reader_writer_class,
                #                                num_foreground_samples_per_case),)))
                for patient, patient_scans in self.patients.items():
                    r.append(p.starmap_async(DatasetFingerprintExtractorLongiSeg.analyze_patient,
                                             ((self.dataset, patient, patient_scans, reader_writer_class,
                                               num_foreground_samples_per_case, preprocessed_output_folder),)))

                remaining = list(range(len(self.patients.keys())))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(desc=None, total=len(self.patients.keys()), disable=self.verbose) as pbar:
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
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractorLongiSeg.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            shapes_after_crop = [e for r in results for e in r[0]]
            spacings = [e for r in results for e in r[1]]
            foreground_intensities_per_channel = [np.concatenate([e[i] for r in results for e in r[2]]) for i in
                                                  range(len(results[0][2][0]))]
            foreground_intensities_per_channel = np.array(foreground_intensities_per_channel)
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [e for r in results for e in r[3]]
            median_relative_size_after_cropping = np.median([e for r in results for e in r[4]], 0)
            num_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
            intensity_statistics_per_channel = {}
            percentiles = np.array((0.5, 50.0, 99.5))
            for i in range(num_channels):
                percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel[i],
                                                                         percentiles)
                intensity_statistics_per_channel[i] = {
                    'mean': float(np.mean(foreground_intensities_per_channel[i])),
                    'median': float(median),
                    'std': float(np.std(foreground_intensities_per_channel[i])),
                    'min': float(np.min(foreground_intensities_per_channel[i])),
                    'max': float(np.max(foreground_intensities_per_channel[i])),
                    'percentile_99_5': float(percentile_99_5),
                    'percentile_00_5': float(percentile_00_5),
                }

            fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint