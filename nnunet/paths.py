#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

"""
Put your personal paths in here. This file will shortly be added to gitignore so that your personal paths will not be tracked
"""
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# You need to set the following folders: base, preprocessing_output_dir and network_training_output_dir. See below for details.


# do not modify these unless you know what you are doing
my_output_identifier = "nnUNetV2"
default_plans_identifier = "nnUNetPlans"
default_data_identifier = 'nnUNet'

try:
    # base is the folder where the raw data is stored. You just need to set base only, the others will be created
    # automatically (they are subfolders of base).
    # Here I use environment variables to set the base folder. Environment variables allow me to use the same code on
    # different systems (and our compute cluster)
    base = os.environ['nnUNet_base']
    raw_dataset_dir = join(base, "nnUNet_raw")
    splitted_4d_output_dir = join(base, "nnUNet_raw_splitted")
    cropped_output_dir = join(base, "nnUNet_raw_cropped")
    maybe_mkdir_p(splitted_4d_output_dir)
    maybe_mkdir_p(raw_dataset_dir)
    maybe_mkdir_p(raw_dataset_dir)
except KeyError:
    cropped_output_dir = splitted_4d_output_dir = raw_dataset_dir = base = None

# preprocessing_output_dir is where the preprocessed data is stored. If you run a training I very strongly recommend
# this is a SSD!
try:
    preprocessing_output_dir = os.environ['nnUNet_preprocessed']
except KeyError:
    preprocessing_output_dir = None

# This is where the trained model parameters are stored
network_training_output_dir = os.path.join(os.environ['RESULTS_FOLDER'], my_output_identifier)
maybe_mkdir_p(network_training_output_dir)
