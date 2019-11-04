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

import shutil

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles

if __name__ == "__main__":
    indir = "/home/fabian/drives/datasets/results/nnUNetOutput_final/predicted_test_sets/Task17_AbdominalOrganSegmentation/ensemble_3d_fullres_cascade_and_3d_fullres"
    outdir = "/home/fabian/drives/datasets/results/nnUNetOutput_final/predicted_test_sets/Task17_AbdominalOrganSegmentation/submit"
    files = subfiles(indir, suffix='nii.gz', prefix="img", join=False)
    maybe_mkdir_p(outdir)
    for f in files:
        outname = "label" + f[3:]
        shutil.copy(join(indir, f), join(outdir, outname))
