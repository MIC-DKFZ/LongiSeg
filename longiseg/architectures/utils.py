from typing import Union
import importlib
import inspect
import pkgutil
from os.path import join
import pydoc

from difference_weighting.utils import recursive_find_python_class

import longiseg


def get_autopet_network_from_plans(arch_class_name, arch_backbone_class_name, arch_kwargs, arch_kwargs_req_import, 
                            input_channels, output_channels, allow_init=True, deep_supervision: Union[bool, None] = None):
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = recursive_find_python_class(join(longiseg.__path__[0], "architectures"), arch_class_name,
                                           "longiseg.architectures")
    if nw_class is None:
        raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        backbone_class_name = arch_backbone_class_name,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network