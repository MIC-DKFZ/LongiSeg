from typing import List
import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, load_json

from longiseg.training.dataloading.longi_dataset import LongiSegDatasetBlosc2


class LongiSegDatasetTracking(LongiSegDatasetBlosc2):
    def load_case(self, patient):
        dparams = {
            'nthreads': 1
        }

        tracking = load_json(join(self.source_folder, f"{patient}.json"))

        if isinstance(tracking, list):
            # if there are multiple scan pairs per patient, we randomly select one of them for training
            tracking = np.random.choice(tracking)

        bl_lesion = int(np.random.choice(list(tracking.keys())))
        bl_point = tracking[str(bl_lesion)]["bl_point"]
        bl_img = tracking[str(bl_lesion)]["img_bl"]

        fu_lesions = tracking[str(bl_lesion)]["merged_lesions"]
        fu_point_prop = tracking[str(bl_lesion)]["fu_point_prop"]
        fu_point = tracking[str(bl_lesion)]["fu_point"]
        fu_img = tracking[str(bl_lesion)]["img_fu"]

        current_data_b2nd_file = join(self.source_folder, fu_img + '.b2nd')
        data_current = blosc2.open(urlpath=current_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_data_b2nd_file = join(self.source_folder, bl_img + '.b2nd')
        data_prior = blosc2.open(urlpath=prior_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        current_seg_b2nd_file = join(self.source_folder, fu_img + '_seg.b2nd')
        seg_current = blosc2.open(urlpath=current_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_seg_b2nd_file = join(self.source_folder, bl_img + '_seg.b2nd')
        seg_prior = blosc2.open(urlpath=prior_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties_fu = load_pickle(join(self.source_folder, fu_img + '.pkl'))
        properties_bl = load_pickle(join(self.source_folder, bl_img + '.pkl'))
        properties_fu['all_fu_lesions'] = fu_lesions
        properties_fu['fu_point'] = fu_point
        properties_fu['fu_point_prop'] = fu_point_prop
        properties_bl['bl_lesion'] = bl_lesion
        properties_bl['bl_point'] = bl_point
        return data_current, seg_current, data_prior, seg_prior, seg_prev, properties_fu, properties_bl

    def load_for_inference(self, patient):
        dparams = {
            'nthreads': 1
        }

        tracking = load_json(join(self.source_folder, f"{patient}.json"))

        if isinstance(tracking, list):
            for scan_dict in tracking:
                for bl_lesion in scan_dict.keys():
                    bl_point = scan_dict[str(bl_lesion)]["bl_point"]
                    bl_img = scan_dict[str(bl_lesion)]["img_bl"]
                    fu_lesion = bl_lesion
                    fu_point = scan_dict[str(bl_lesion)]["fu_point_prop"]
                    if np.isnan(fu_point).all():
                        continue
                    fu_img = scan_dict[str(bl_lesion)]["img_fu"]

                    fu_data_b2nd_file = join(self.source_folder, fu_img + '.b2nd')
                    data_fu = blosc2.open(urlpath=fu_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                    bl_data_b2nd_file = join(self.source_folder, bl_img + '.b2nd')
                    data_bl = blosc2.open(urlpath=bl_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                    bl_seg_b2nd_file = join(self.source_folder, bl_img + '_seg.b2nd')
                    seg_bl = blosc2.open(urlpath=bl_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                    properties = load_pickle(join(self.source_folder, fu_img + '.pkl'))
                    properties['bl_lesion'] = int(bl_lesion)
                    properties['fu_lesion'] = int(fu_lesion)
                    properties['bl_point'] = bl_point
                    properties['fu_point'] = fu_point
                    properties['fu_img'] = fu_img

                    yield data_fu, None, data_bl, seg_bl, None, properties

        else:
            for bl_lesion in tracking.keys():
                bl_point = tracking[str(bl_lesion)]["bl_point"]
                bl_img = tracking[str(bl_lesion)]["img_bl"]
                fu_lesion = bl_lesion
                fu_point = tracking[str(bl_lesion)]["fu_point_prop"]
                if np.isnan(fu_point).all():
                    continue
                fu_img = tracking[str(bl_lesion)]["img_fu"]

                np.isnan(tracking[str(fu_lesion)]["fu_point_prop"]).all()

                fu_data_b2nd_file = join(self.source_folder, fu_img + '.b2nd')
                data_fu = blosc2.open(urlpath=fu_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                bl_data_b2nd_file = join(self.source_folder, bl_img + '.b2nd')
                data_bl = blosc2.open(urlpath=bl_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                bl_seg_b2nd_file = join(self.source_folder, bl_img + '_seg.b2nd')
                seg_bl = blosc2.open(urlpath=bl_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

                properties = load_pickle(join(self.source_folder, fu_img + '.pkl'))
                properties['bl_lesion'] = int(bl_lesion)
                properties['fu_lesion'] = int(fu_lesion)
                properties['bl_point'] = bl_point
                properties['fu_point'] = fu_point
                properties['fu_img'] = fu_img

                yield data_fu, None, data_bl, seg_bl, None, properties


class LongiSegDatasetTrackingPretrain(LongiSegDatasetTracking):
    def load_case(self, patient):
        dparams = {
            'nthreads': 1
        }

        fu_data_b2nd_file = join(self.source_folder, self.patients[patient][0] + '.b2nd')
        data_fu = blosc2.open(urlpath=fu_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        bl_data_b2nd_file = join(self.source_folder, self.patients[patient][1] + '.b2nd')
        data_bl = blosc2.open(urlpath=bl_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        fu_seg_b2nd_file = join(self.source_folder, self.patients[patient][0] + '_seg.b2nd')
        seg_fu = blosc2.open(urlpath=fu_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        bl_seg_b2nd_file = join(self.source_folder, self.patients[patient][1] + '_seg.b2nd')
        seg_bl = blosc2.open(urlpath=bl_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties_fu = load_pickle(join(self.source_folder, self.patients[patient][0] + '.pkl'))
        properties_bl = load_pickle(join(self.source_folder, self.patients[patient][1] + '.pkl'))
        return data_fu, seg_fu, data_bl, seg_bl, seg_prev, properties_fu, properties_bl


def infer_dataset_class(folder: str, pretrain: bool = False) -> type:
   return LongiSegDatasetTracking if not pretrain else LongiSegDatasetTrackingPretrain