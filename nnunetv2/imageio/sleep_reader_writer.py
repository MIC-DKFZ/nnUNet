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
        assert len(image_fnames) == 1
        image = np.load(image_fnames[0])  # input shape: (n_samples, 6)
        image = image.T[:, None, :, None]  # output shape: (6, 1, n_samples, 1) as (c, 1, y, z)
        return image.astype(np.float32), {'spacing': (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        seg = np.load(seg_fname)  # input shape: (n_samples,)
        seg = seg[None, None, :, None]  # output shape: (1, 1, n_samples, 1) as (1, 1, y, z)
        return seg.astype(np.uint8), {'spacing': (999, 1, 1)}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        np.save(output_fname, np.squeeze(seg.astype(np.uint8)))
