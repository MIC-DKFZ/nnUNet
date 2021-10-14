import numpy as np


def recursive_fix_for_json_export(my_dict: dict):
    # json is stupid. 'cannot serialize object of type bool_/int64/float64'. Come on bro.
    for k in my_dict.keys():
        if isinstance(my_dict[k], dict):
            recursive_fix_for_json_export(my_dict[k])
        elif isinstance(my_dict[k], np.ndarray):
            assert len(my_dict[k].shape) == 1, 'only 1d arrays are supported'
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=list)
        elif isinstance(my_dict[k], (np.bool_,)):
            my_dict[k] = bool(my_dict[k])
        elif isinstance(my_dict[k], (np.int64, np.int32, np.int8, np.uint8)):
            my_dict[k] = int(my_dict[k])
        elif isinstance(my_dict[k], (np.float32, np.float64, np.float16)):
            my_dict[k] = float(my_dict[k])
        elif isinstance(my_dict[k], (list, tuple)):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=type(my_dict[k]))


def fix_types_iterable(iterable, output_type):
    out = []
    for i in iterable:
        if type(i) in (np.int64, np.int32, np.int8, np.uint8):
            out.append(int(i))
        elif type(i) in (np.float32, np.float64, np.float16):
            out.append(float(i))
        elif type(i) in (np.bool_,):
            out.append(bool(i))
        else:
            out.append(i)
    return output_type(out)