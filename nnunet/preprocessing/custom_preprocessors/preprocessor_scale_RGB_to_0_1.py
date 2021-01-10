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

import numpy as np
from nnunet.preprocessing.preprocessing import PreprocessorFor2D, resample_patient


class GenericPreprocessor_scale_uint8_to_0_1(PreprocessorFor2D):
    """
    For RGB images with a value range of [0, 255]. This preprocessor overwrites the default normalization scheme by
    normalizing intensity values through a simple division by 255 which rescales them to [0, 1]

    NOTE THAT THIS INHERITS FROM PreprocessorFor2D, SO ITS WRITTEN FOR 2D ONLY! WHEN CREATING A PREPROCESSOR FOR 3D
    DATA, USE GenericPreprocessor AS PARENT!
    """
    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        ############ THIS PART IS IDENTICAL TO PARENT CLASS ################

        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        ############ HERE IS WHERE WE START CHANGING THINGS!!!!!!!################

        # this is where the normalization takes place. We ignore use_nonzero_mask and normalization_scheme_per_modality
        for c in range(len(data)):
            data[c] = data[c].astype(np.float32) / 255.
        return data, seg, properties