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
__paths = {
    "nnUNet_raw": os.environ.get("nnUNet_raw"),
    "nnUNet_preprocessed": os.environ.get("nnUNet_preprocessed"),
    "nnUNet_results": os.environ.get("nnUNet_results"),
}


def set_paths(
    nnUNet_raw: str | None = None,
    nnUNet_preprocessed: str | None = None,
    nnUNet_results: str | None = None,
):
    if nnUNet_raw is not None:
        __paths["nnUNet_raw"] = nnUNet_raw
    if nnUNet_preprocessed is not None:
        __paths["nnUNet_preprocessed"] = nnUNet_preprocessed
    if nnUNet_results is not None:
        __paths["nnUNet_results"] = nnUNet_results


def __getattr__(item: str):
    if item == "nnUNet_raw":
        if __paths["nnUNet_raw"] is None:
            print(
                "nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
                "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
                "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
                "this up properly."
            )
        return __paths["nnUNet_raw"]
    elif item == "nnUNet_preprocessed":
        if __paths["nnUNet_preprocessed"] is None:
            print(
                "nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
                "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
                "to set this up."
            )
        return __paths["nnUNet_preprocessed"]
    elif item == "nnUNet_results":
        if __paths["nnUNet_results"] is None:
            print(
                "nnUNet_results is not defined and nnU-Net cannot be used for training or "
                "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
                "on how to set this up."
            )
        return __paths["nnUNet_results"]
    else:
        raise AttributeError(f"module {__name__} has no attribute {item}")
