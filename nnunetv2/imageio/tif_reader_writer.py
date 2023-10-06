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
import os.path
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import tifffile
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json, save_json, split_path, join


class Tiff3DIO(BaseReaderWriter):
    """
    reads and writes 3D tif(f) images. Uses tifffile package. Ignores metadata (for now)!

    If you have 2D tiffs, use NaturalImage2DIO

    Supports the use of auxiliary files for spacing information. If used, the auxiliary files are expected to end
    with .json and omit the channel identifier. So, for example, the corresponding of image image1_0000.tif is
    expected to be image1.json)!
    """
    supported_file_endings = [
        '.tif',
        '.tiff',
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        # figure out file ending used here
        ending = '.' + image_fnames[0].split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'
        ending_length = len(ending)
        truncate_length = ending_length + 5 # 5 comes from len(_0000)

        images = []
        for f in image_fnames:
            image = tifffile.imread(f)
            if len(image.shape) != 3:
                raise RuntimeError(f"Only 3D images are supported! File: {f}")
            images.append(image[None])

        # see if aux file can be found
        expected_aux_file = image_fnames[0][:-truncate_length] + '.json'
        if isfile(expected_aux_file):
            spacing = load_json(expected_aux_file)['spacing']
            assert len(spacing) == 3, f'spacing must have 3 entries, one for each dimension of the image. File: {expected_aux_file}'
        else:
            print(f'WARNING no spacing file found for images {image_fnames}\nAssuming spacing (1, 1, 1).')
            spacing = (1, 1, 1)

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        return np.vstack(images).astype(np.float32), {'spacing': spacing}

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # not ideal but I really have no clue how to set spacing/resolution information properly in tif files haha
        tifffile.imwrite(output_fname, data=seg.astype(np.uint8), compression='zlib')
        file = os.path.basename(output_fname)
        out_dir = os.path.dirname(output_fname)
        ending = file.split('.')[-1]
        save_json({'spacing': properties['spacing']}, join(out_dir, file[:-(len(ending) + 1)] + '.json'))

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        # figure out file ending used here
        ending = '.' + seg_fname.split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'
        ending_length = len(ending)

        seg = tifffile.imread(seg_fname)
        if len(seg.shape) != 3:
            raise RuntimeError(f"Only 3D images are supported! File: {seg_fname}")
        seg = seg[None]

        # see if aux file can be found
        expected_aux_file = seg_fname[:-ending_length] + '.json'
        if isfile(expected_aux_file):
            spacing = load_json(expected_aux_file)['spacing']
            assert len(spacing) == 3, f'spacing must have 3 entries, one for each dimension of the image. File: {expected_aux_file}'
            assert all([i > 0 for i in spacing]), f"Spacing must be > 0, spacing: {spacing}"
        else:
            print(f'WARNING no spacing file found for segmentation {seg_fname}\nAssuming spacing (1, 1, 1).')
            spacing = (1, 1, 1)

        return seg.astype(np.float32), {'spacing': spacing}