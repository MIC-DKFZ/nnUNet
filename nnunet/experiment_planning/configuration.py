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

# experiment planning

# a class in a patient will be set to background if it has less than X times the volume of the minimum volume of
# that class in the training data
MIN_SIZE_PER_CLASS_FACTOR = 0.5  # not used, ignore this
TARGET_SPACING_PERCENTILE = 50

FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK = 4
FEATUREMAP_MIN_EDGE_LENGTH_BOTTLENECK2 = 6  # not used, ignore this

RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD = 3  # z is defined as the axis with the highest spacing, also used to
# determine whether we use 2d or 3d data augmentation

HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 = 4  # 1/4 of a patient

batch_size_covers_max_percent_of_dataset = 0.05 # all samples in the batch together cannot cover more than 5% of the entire dataset
dataset_min_batch_size_cap = 2 # if the dataset size dictates a very small batch size, do not make that smaller than 3 (if architecture dictates smaller batch size then use the smaller one of these two)



