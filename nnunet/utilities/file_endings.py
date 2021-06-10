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


from batchgenerators.utilities.file_and_folder_operations import *


def remove_trailing_slash(filename: str):
    return os.path.normpath(filename)


def get_last_folder(foldername: str):
    assert os.path.isdir(foldername), "Provided argument 'foldername' is not a folder!"
    return os.path.basename(remove_trailing_slash(foldername))


def maybe_add_0000_to_all_niigz(folder):
    nii_gz = subfiles(folder, suffix='.nii.gz')
    for n in nii_gz:
        n = remove_trailing_slash(n)
        if not n.endswith('_0000.nii.gz'):
            os.rename(n, n[:-7] + '_0000.nii.gz')
