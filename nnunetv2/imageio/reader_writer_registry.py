from typing import Type

from batchgenerators.utilities.file_and_folder_operations import join

import nnunetv2
from nnunetv2.imageio.natural_image_reager_writer import NaturalImage2DIO
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.tif_reader_writer import Tiff3DIO
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

LIST_OF_IO_CLASSES = [
    NaturalImage2DIO,
    SimpleITKIO,
    Tiff3DIO,
    NibabelIO
]


def determine_reader_writer(dataset_json_content: dict, example_file: str = None) -> Type[BaseReaderWriter]:
    if 'overwrite_image_reader_writer' in dataset_json_content.keys() and \
            dataset_json_content['overwrite_image_reader_writer'] != 'None':
        ioclass_name = dataset_json_content['overwrite_image_reader_writer']
        # trying to find that class in the nnunetv2.imageio module
        ret = recursive_find_python_class(join(nnunetv2.__path__[0], "imageio"), ioclass_name, 'nnunetv2.imageio')
        if ret is None:
            print('Warning: Unable to find ioclass specified in dataset.json: %s' % ioclass_name)
            print('Trying to automatically determine desired class')
        else:
            print('Using %s reader/writer' % ret)
            return ret
    return auto_find_reader_writer(dataset_json_content['file_ending'], example_file)


def auto_find_reader_writer(file_ending: str, file: str = None):
    for rw in LIST_OF_IO_CLASSES:
        if file_ending in rw.supported_file_endings:
            if file is not None:
                # if an example file is provided, try if we can actually read it. If not move on to the next reader
                try:
                    tmp = rw()
                    _ = tmp.read_images((file,))
                    print('Using %s as reader/writer' % rw)
                    return rw
                except:
                    pass
            else:
                print('Using %s as reader/writer' % rw)
                return rw
    raise RuntimeError("Unable to determine a reader for file ending %s and file %s (file None means no file provided)." % (file_ending, file))
