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
PLEASE READ documentation/setting_up_paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""


class _EnvPath(os.PathLike):
    def __init__(self, env_var_name: str, missing_message: str):
        self.env_var_name = env_var_name
        self.missing_message = missing_message

    def get(self):
        return os.environ.get(self.env_var_name)

    def is_set(self) -> bool:
        return self.get() is not None

    def require(self) -> str:
        value = self.get()
        if value is None:
            raise RuntimeError(
                f"{self.env_var_name} is not defined. {self.missing_message} "
                f"Please read documentation/setting_up_paths.md for information on how to set this up."
            )
        return value

    def __fspath__(self) -> str:
        return self.require()

    def __str__(self) -> str:
        return self.require()

    def __repr__(self) -> str:
        value = self.get()
        return repr(value) if value is not None else f"<unset {self.env_var_name}>"

    def __bool__(self) -> bool:
        return self.is_set()

    def __eq__(self, other) -> bool:
        if other is None:
            return self.get() is None
        return self.get() == other


nnUNet_raw = _EnvPath(
    'nnUNet_raw',
    "nnU-Net can only be used on data for which preprocessed files are already present on your system. "
    "nnU-Net cannot be used for experiment planning and preprocessing like this. If this is not intended, "
)
nnUNet_preprocessed = _EnvPath(
    'nnUNet_preprocessed',
    "nnU-Net cannot be used for preprocessing or training. If this is not intended, "
)
nnUNet_results = _EnvPath(
    'nnUNet_results',
    "nnU-Net cannot be used for training or inference. If this is not intended behavior, "
)
nnUNet_extTrainer = _EnvPath(
    'nnUNet_extTrainer',
    "nnU-Net cannot locate custom trainer classes from external directories. If this is not intended, "
)
