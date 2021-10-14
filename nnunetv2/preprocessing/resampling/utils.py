from typing import Callable

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def recursive_find_resampling_nf_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "resampling"), resampling_fn,
                                      'nnunetv2.preprocessing.resampling')
    if ret is None:
        raise RuntimeError("Unable to find resampling function named '%s'. Please make sure this fn is located in the "
                           "nnunetv2.preprocessing.resampling module." % resampling_fn)
    else:
        return ret
