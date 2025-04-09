#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

LongiSeg_raw = os.environ.get('LongiSeg_raw')
LongiSeg_preprocessed = os.environ.get('LongiSeg_preprocessed')
LongiSeg_results = os.environ.get('LongiSeg_results')

if LongiSeg_raw is None:
    print("Could not find LongiSeg_raw environment variable, falling back to nnUNet_raw")
    LongiSeg_raw = os.environ.get('nnUNet_raw')
if LongiSeg_preprocessed is None:
    print("Could not find LongiSeg_preprocessed environment variable, falling back to nnUNet_preprocessed")
    LongiSeg_preprocessed = os.environ.get('nnUNet_preprocessed')
if LongiSeg_results is None:
    print("Could not find LongiSeg_results environment variable, falling back to nnUNet_results")
    LongiSeg_results = os.environ.get('nnUNet_results')

if LongiSeg_raw is None:
    print("LongiSeg_raw is not defined and LongiSeg can only be used on data for which preprocessed files "
          "are already present on your system. LongiSeg cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if LongiSeg_preprocessed is None:
    print("LongiSeg_preprocessed is not defined and LongiSeg can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if LongiSeg_results is None:
    print("LongiSeg_results is not defined and LongiSeg cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
