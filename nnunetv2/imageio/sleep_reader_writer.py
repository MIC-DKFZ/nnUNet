#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
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

from typing import Tuple, Union, List

import numpy as np

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class SleepReaderWriter(BaseReaderWriter):
    supported_file_endings = ['.npy']

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        image_list = []
        for f in image_fnames:
            image = np.load(f)  # input shape (n_samples, 6)
            image = image.T[None, :, :, None]  # to shape (6, 1, n_samples, 1) as (c, 1, y, z)
            image_list.append(image)
        assert self._check_all_same([i.shape for i in image_list])
        return np.vstack(image_list).astype(np.float32), {'spacing': (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        seg_list = []
        for f in seg_fname:
            seg = np.load(f)  # input shape (n_samples,)
            seg_list.append(seg[None, None, :, None])  # to shape (1, 1, n_samples, 1) as (1, 1, y, z)
        assert self._check_all_same([i.shape for i in seg_list])
        return np.vstack(seg_list).astype(np.uint8), {'spacing': (999, 1, 1)}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        np.save(output_fname, np.squeeze(seg.astype(np.uint8)))
