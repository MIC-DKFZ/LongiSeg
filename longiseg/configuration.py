import os

from longiseg.utilities.default_n_proc_DA import get_allowed_n_proc_DA

if 'LongiSeg_def_n_proc' in os.environ:
    default_num_processes = int(os.environ['LongiSeg_def_n_proc'])
elif 'nnUNet_def_n_proc' in os.environ:
    default_num_processes = int(os.environ['nnUNet_def_n_proc'])
else:
    default_num_processes = 8

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()
