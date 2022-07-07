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
import subprocess
import os


def get_allowed_n_proc_DA():
    hostname = subprocess.getoutput(['hostname'])

    if 'nnUNet_n_proc_DA' in os.environ.keys():
        return int(os.environ['nnUNet_n_proc_DA'])

    if hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'e230-AMDworkstation']:
        return 16

    if hostname in ['Fabian',]:
        return 12

    if hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
        return 12
    elif hostname.startswith('e230-dgx1'):
        return 10
    elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
        return 16
    elif hostname.startswith('e230-dgx2'):
        return 6
    elif hostname.startswith('e230-dgxa100-'):
        return 28
    elif hostname.startswith('lsf22-gpu'):
        return 28
    else:
        return None
