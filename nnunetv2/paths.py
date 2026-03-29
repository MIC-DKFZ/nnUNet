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
import warnings

"""
PLEASE READ documentation/setting_up_paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnUNet_raw = os.environ.get('nnUNet_raw')
nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

_SETTING_UP_PATHS_DOC = 'documentation/setting_up_paths.md'
_DEFAULT_USAGE_BY_ENV_VAR = {
    'nnUNet_raw': 'experiment planning and preprocessing',
    'nnUNet_preprocessed': 'preprocessing and training',
    'nnUNet_results': 'training and inference',
}


def _build_missing_path_message(env_var_name: str, required_for: str = None) -> str:
    message = f"Environment variable '{env_var_name}' is not set."
    if required_for is not None:
        message += f" It is required for {required_for}."
    message += f" Please configure it according to {_SETTING_UP_PATHS_DOC}"
    return message


def _warn_if_missing(env_var_name: str, current_value: str) -> None:
    if current_value is None:
        warnings.warn(
            _build_missing_path_message(env_var_name, _DEFAULT_USAGE_BY_ENV_VAR[env_var_name]),
            stacklevel=1,
        )


def get_required_path(env_var_name: str, required_for: str = None) -> str:
    value = os.environ.get(env_var_name)
    if value is None:
        raise EnvironmentError(_build_missing_path_message(env_var_name, required_for))
    return value


def require_raw_dataset_path(required_for: str = None) -> str:
    return get_required_path('nnUNet_raw', required_for)


def require_preprocessed_dataset_path(required_for: str = None) -> str:
    return get_required_path('nnUNet_preprocessed', required_for)


def require_results_path(required_for: str = None) -> str:
    return get_required_path('nnUNet_results', required_for)


_warn_if_missing('nnUNet_raw', nnUNet_raw)
_warn_if_missing('nnUNet_preprocessed', nnUNet_preprocessed)
_warn_if_missing('nnUNet_results', nnUNet_results)
