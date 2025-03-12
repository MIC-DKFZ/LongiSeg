from typing import Union, List, Tuple

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
import torch


class MergeTransform(BasicTransform):
    def apply(self, data_dict, **params):
        if data_dict.get('image_current') is not None and data_dict.get('image_prior') is not None:
            data_dict['image'] = self._apply_to_tensor(data_dict['image_current'], data_dict['image_prior'], **params)
            del data_dict['image_current'], data_dict['image_prior']
        else:
            raise RuntimeError("MergeTransform requires 'image_current' and 'image_prior' in data_dict")
        if data_dict.get('segmentation_current') is not None and data_dict.get('segmentation_prior') is not None:
            data_dict['segmentation'] = self._apply_to_tensor(data_dict['segmentation_current'], data_dict['segmentation_prior'], **params)
            del data_dict['segmentation_current'], data_dict['segmentation_prior']
        else:
            raise RuntimeError("MergeTransform requires 'segmentation_current' and 'segmentation_prior' in data_dict")
        return data_dict
    
    def _apply_to_tensor(self, current: torch.Tensor, prior: torch.Tensor, **params) -> torch.Tensor:
        merged = torch.cat([current, prior], dim=0)
        return merged


class SplitTransform(BasicTransform):
    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image_current'], data_dict['image_prior'] = self._apply_to_tensor(data_dict['image'], **params)
            del data_dict['image']
        else:
            raise RuntimeError("SplitTransform requires 'image' in data_dict")
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation_current'], data_dict['segmentation_prior'] = self._apply_to_tensor(data_dict['segmentation'], **params)
            del data_dict['segmentation']
        else:
            raise RuntimeError("SplitTransform requires 'segmentation' in data_dict")
        return data_dict
    
    def _apply_to_tensor(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        channels = tensor.shape[0]
        current = tensor[:channels//2]
        prior = tensor[channels//2:]
        return current, prior
    

class ConvertSegToOneHot(BasicTransform):
    def __init__(self, foreground_labels: Union[Tuple[int, ...], List[int]], key: str = "segmentation_prior", 
                 dtype_key: str = "image_prior"):
        super().__init__()
        self.foreground_labels = foreground_labels
        self.key = key
        self.dtype_key = dtype_key

    def apply(self, data_dict, **params):
        seg = data_dict[self.key]
        seg_onehot = torch.zeros((len(self.foreground_labels), *seg.shape), dtype=data_dict[self.dtype_key].dtype)
        for i, l in enumerate(self.foreground_labels):
            seg_onehot[i][seg == l] = 1
        data_dict[self.key] = seg_onehot.squeeze_(1)
        return data_dict


class DownsampleSegForDSTransformLongi(DownsampleSegForDSTransform):
    def __init__(self, ds_scales: Union[List, Tuple], key: str = "segmentation_current"):
        super().__init__(ds_scales)
        self.key = key

    def apply(self, data_dict, **params):
        data_dict[self.key] = self._apply_to_segmentation(data_dict[self.key], **params)
        return data_dict