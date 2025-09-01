from typing import List
import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, load_json

from longiseg.training.dataloading.longi_dataset import LongiSegDatasetBlosc2


class AutoPETDataset(LongiSegDatasetBlosc2):
    def load_case(self, patient):
        dparams = {
            'nthreads': 1
        }

        tracking = load_json(join(self.source_folder, f"{patient}.json"))

        bl_lesion = int(np.random.choice(list(tracking.keys())))
        bl_point = tracking[str(bl_lesion)]["bl_point"]
        bl_img = tracking[str(bl_lesion)]["img_bl"]

        fu_lesions = tracking[str(bl_lesion)]["merged_lesions"]
        fu_lesion = np.random.choice(fu_lesions)
        if fu_lesion == 0:
            fu_point = tracking[str(bl_lesion)]["fu_point_prop"]
        elif np.isnan(tracking[str(fu_lesion)]["fu_point_prop"]).all():
            fu_point = tracking[str(fu_lesion)]["fu_point"]
        elif np.isnan(tracking[str(fu_lesion)]["fu_point"]).all():
            fu_point = tracking[str(fu_lesion)]["fu_point_prop"]
        elif np.random.rand() < 0.5:
            fu_point = tracking[str(fu_lesion)]["fu_point_prop"]
        else:
            fu_point = tracking[str(fu_lesion)]["fu_point"]
        fu_img = tracking[str(bl_lesion)]["img_fu"]

        current = f"{patient}_FU_img_{fu_img:02d}"
        prior = f"{patient}_BL_img_{bl_img:02d}"

        current_data_b2nd_file = join(self.source_folder, current + '.b2nd')
        data_current = blosc2.open(urlpath=current_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_data_b2nd_file = join(self.source_folder, prior + '.b2nd')
        data_prior = blosc2.open(urlpath=prior_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        current_seg_b2nd_file = join(self.source_folder, current + '_seg.b2nd')
        seg_current = blosc2.open(urlpath=current_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_seg_b2nd_file = join(self.source_folder, prior + '_seg.b2nd')
        seg_prior = blosc2.open(urlpath=prior_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties = load_pickle(join(self.source_folder, current + '.pkl'))
        properties['bl_lesion'] = bl_lesion
        properties['fu_lesion'] = fu_lesion
        properties['all_fu_lesions'] = fu_lesions
        properties['bl_point'] = bl_point
        properties['fu_point'] = fu_point
        return data_current, seg_current, data_prior, seg_prior, seg_prev, properties

    def load_for_inference(self, patient):
        dparams = {
            'nthreads': 1
        }

        tracking = load_json(join(self.source_folder, f"{patient}.json"))

        for bl_lesion in tracking.keys():
            bl_point = tracking[str(bl_lesion)]["bl_point"]
            bl_img = tracking[str(bl_lesion)]["img_bl"]
            fu_lesion = bl_lesion
            fu_point = tracking[str(bl_lesion)]["fu_point_prop"]
            if np.isnan(fu_point).all():
                continue
            fu_img = tracking[str(bl_lesion)]["img_fu"]

            np.isnan(tracking[str(fu_lesion)]["fu_point_prop"]).all()

            current = f"{patient}_FU_img_{fu_img:02d}"
            prior = f"{patient}_BL_img_{bl_img:02d}"

            current_data_b2nd_file = join(self.source_folder, current + '.b2nd')
            data_current = blosc2.open(urlpath=current_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

            prior_data_b2nd_file = join(self.source_folder, prior + '.b2nd')
            data_prior = blosc2.open(urlpath=prior_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

            prior_seg_b2nd_file = join(self.source_folder, prior + '_seg.b2nd')
            seg_prior = blosc2.open(urlpath=prior_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

            properties = load_pickle(join(self.source_folder, current + '.pkl'))
            properties['bl_lesion'] = int(bl_lesion)
            properties['fu_lesion'] = int(fu_lesion)
            properties['bl_point'] = bl_point
            properties['fu_point'] = fu_point
            properties['current'] = current

            yield data_current, None, data_prior, seg_prior, None, properties


class AutoPETDatasetPretrain(AutoPETDataset):
    def load_case(self, patient):
        dparams = {
            'nthreads': 1
        }

        current_data_b2nd_file = join(self.source_folder, self.patients[patient][0] + '.b2nd')
        data_current = blosc2.open(urlpath=current_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_data_b2nd_file = join(self.source_folder, self.patients[patient][1] + '.b2nd')
        data_prior = blosc2.open(urlpath=prior_data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        current_seg_b2nd_file = join(self.source_folder, self.patients[patient][0] + '_seg.b2nd')
        seg_current = blosc2.open(urlpath=current_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        prior_seg_b2nd_file = join(self.source_folder, self.patients[patient][1] + '_seg.b2nd')
        seg_prior = blosc2.open(urlpath=prior_seg_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties_current = load_pickle(join(self.source_folder, self.patients[patient][0] + '.pkl'))
        properties_prior = load_pickle(join(self.source_folder, self.patients[patient][1] + '.pkl'))
        return data_current, seg_current, data_prior, seg_prior, seg_prev, properties_current, properties_prior


def infer_dataset_class(folder: str, pretrain: bool = False) -> type:
   return AutoPETDataset if not pretrain else AutoPETDatasetPretrain