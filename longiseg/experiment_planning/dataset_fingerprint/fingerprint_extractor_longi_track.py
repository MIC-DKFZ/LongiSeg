from typing import Type

from longiseg.imageio.base_reader_writer import BaseReaderWriter

from longiseg.experiment_planning.dataset_fingerprint.fingerprint_extractor_longi import \
    DatasetFingerprintExtractorLongiSeg


class DatasetFingerprintExtractorLongiSegTrack(DatasetFingerprintExtractorLongiSeg):
    @staticmethod
    def analyze_patient(dataset: dict, patient: str, patient_scans: list, reader_writer_class: Type[BaseReaderWriter],
                        num_samples: int = 10000, preprocess_output_folder: str = None):
        num_sample_patient = num_samples // len(patient_scans)
        shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
        relative_size_after_cropping = [], [], [], [], []
        for s in patient_scans:
            image_files = dataset[s]['images']
            segmentation_file = dataset[s]['label']
            case_analysis = DatasetFingerprintExtractorLongiSegTrack.analyze_case(image_files, segmentation_file,
                                                                        reader_writer_class, num_sample_patient)
            shape_after_crop.append(case_analysis[0])
            spacing.append(case_analysis[1])
            foreground_intensities_per_channel.append(case_analysis[2])
            foreground_intensity_stats_per_channel.append(case_analysis[3])
            relative_size_after_cropping.append(case_analysis[4])
        return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
                relative_size_after_cropping
