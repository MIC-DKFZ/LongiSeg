import pydoc
import warnings
from os.path import join
from torch import nn
import torch

import dynamic_network_architectures

from difference_weighting.utils import recursive_find_python_class


class LongiUNetAutoPET(nn.Module):
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__()

        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=2*input_channels+3*(num_classes-1),
            num_classes=num_classes,
            **architecture_kwargs
        )

    def forward(self, d_c, d_p=None, t_p=None, g_c=None, g_p=None):
        # allow for concatenation at different points in the code
        if d_p is None and t_p is None and g_c is None and g_p is None:
            x = d_c
        else:
            x = torch.cat([d_c, d_p, t_p, g_c, g_p], dim=1)
        return self.backbone(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


class CrossSecUNetPointAutoPET(LongiUNetAutoPET):
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__(input_channels, num_classes, backbone_class_name, **architecture_kwargs)
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels+num_classes-1,
            num_classes=num_classes,
            **architecture_kwargs
        )

    def forward(self, d_c, d_p=None, t_p=None, g_c=None, g_p=None):
        # allow for concatenation at different points in the code
        if d_p is None and t_p is None and g_c is None and g_p is None:
            d_c, d_p, t_p, g_c, g_p = torch.tensor_split(d_c, 5, dim=1)
        x = torch.cat([d_c, g_c], dim=1)
        return self.backbone(x)


class CrossSecUNetMaskAutoPET(LongiUNetAutoPET):
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__(input_channels, num_classes, backbone_class_name, **architecture_kwargs)
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels+2*(num_classes-1),
            num_classes=num_classes,
            **architecture_kwargs
        )

    def forward(self, d_c, d_p=None, t_p=None, g_c=None, g_p=None):
        # allow for concatenation at different points in the code
        if d_p is None and t_p is None and g_c is None and g_p is None:
            d_c, d_p, t_p, g_c, g_p = torch.tensor_split(d_c, 5, dim=1)
        x = torch.cat([d_c, t_p, g_c], dim=1)
        return self.backbone(x)