import numpy as np
import blosc2

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, load_json

from longiseg.training.dataloading.longi_dataset import LongiSegDatasetBlosc2


class LongiSegDatasetTracking(LongiSegDatasetBlosc2):
    dparams = {'nthreads': 1}

    def _open(self, identifier: str, seg: bool = False):
        suffix = '_seg.b2nd' if seg else '.b2nd'
        return blosc2.open(urlpath=join(self.source_folder, identifier + suffix), mode='r', dparams=self.dparams,
                           mmap_mode='r')

    def _load_tracking(self, patient) -> list:
        tracking = load_json(join(self.source_folder, f"{patient}.json"))
        return tracking if isinstance(tracking, list) else [tracking]

    def load_case(self, patient):
        tracking = self._load_tracking(patient)
        # if there are multiple scan pairs per patient, we randomly select one of them for training
        tracking = tracking[np.random.randint(len(tracking))]

        bl_lesion = int(np.random.choice(list(tracking.keys())))
        lesion_info = tracking[str(bl_lesion)]
        bl_img = lesion_info["img_bl"]
        fu_img = lesion_info["img_fu"]

        data_current = self._open(fu_img)
        data_prior = self._open(bl_img)
        seg_current = self._open(fu_img, seg=True)
        seg_prior = self._open(bl_img, seg=True)

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties_fu = load_pickle(join(self.source_folder, fu_img + '.pkl'))
        properties_bl = load_pickle(join(self.source_folder, bl_img + '.pkl'))
        properties_fu['all_fu_lesions'] = lesion_info["merged_lesions"]
        properties_fu['fu_point'] = lesion_info["fu_point"]
        properties_fu['fu_point_prop'] = lesion_info["fu_point_prop"]
        properties_bl['bl_lesion'] = bl_lesion
        properties_bl['bl_point'] = lesion_info["bl_point"]
        return data_current, seg_current, data_prior, seg_prior, seg_prev, properties_fu, properties_bl

    def load_for_inference(self, patient):
        for scan_dict in self._load_tracking(patient):
            for bl_lesion, lesion_info in scan_dict.items():
                fu_point = lesion_info["fu_point_prop"]
                if np.isnan(fu_point).all():
                    continue
                bl_img = lesion_info["img_bl"]
                fu_img = lesion_info["img_fu"]

                data_fu = self._open(fu_img)
                data_bl = self._open(bl_img)
                seg_bl = self._open(bl_img, seg=True)

                properties = load_pickle(join(self.source_folder, fu_img + '.pkl'))
                properties['bl_lesion'] = int(bl_lesion)
                properties['fu_lesion'] = int(bl_lesion)
                properties['bl_point'] = lesion_info["bl_point"]
                properties['fu_point'] = fu_point
                properties['fu_img'] = fu_img

                yield data_fu, None, data_bl, seg_bl, None, properties


class LongiSegDatasetTrackingPretrain(LongiSegDatasetTracking):
    def load_case(self, patient):
        fu_img, bl_img = self.patients[patient][0], self.patients[patient][1]

        data_fu = self._open(fu_img)
        data_bl = self._open(bl_img)
        seg_fu = self._open(fu_img, seg=True)
        seg_bl = self._open(bl_img, seg=True)

        if self.folder_with_segs_from_previous_stage is not None:
            raise NotImplementedError("Cascade is not implemented for longitudinal segmentation")
        else:
            seg_prev = None

        properties_fu = load_pickle(join(self.source_folder, fu_img + '.pkl'))
        properties_bl = load_pickle(join(self.source_folder, bl_img + '.pkl'))
        return data_fu, seg_fu, data_bl, seg_bl, seg_prev, properties_fu, properties_bl


def infer_dataset_class(folder: str, pretrain: bool = False) -> type:
    return LongiSegDatasetTrackingPretrain if pretrain else LongiSegDatasetTracking
