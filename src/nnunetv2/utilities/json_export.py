from collections.abc import Iterable

import numpy as np
import torch


def recursive_fix_for_json_export(my_dict: dict):
    # json is ... a very nice thing to have
    # 'cannot serialize object of type bool_/int64/float64'. Apart from that of course...
    keys = list(my_dict.keys())  # cannot iterate over keys() if we change keys....
    for k in keys:
        if isinstance(k, (np.int64, np.int32, np.int8, np.uint8)):
            tmp = my_dict[k]
            del my_dict[k]
            my_dict[int(k)] = tmp
            del tmp
            k = int(k)

        if isinstance(my_dict[k], dict):
            recursive_fix_for_json_export(my_dict[k])
        elif isinstance(my_dict[k], np.ndarray):
            assert my_dict[k].ndim == 1, 'only 1d arrays are supported'
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=list)
        elif isinstance(my_dict[k], (np.bool_,)):
            my_dict[k] = bool(my_dict[k])
        elif isinstance(my_dict[k], (np.int64, np.int32, np.int8, np.uint8)):
            my_dict[k] = int(my_dict[k])
        elif isinstance(my_dict[k], (np.float32, np.float64, np.float16)):
            my_dict[k] = float(my_dict[k])
        elif isinstance(my_dict[k], list):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=type(my_dict[k]))
        elif isinstance(my_dict[k], tuple):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=tuple)
        elif isinstance(my_dict[k], torch.device):
            my_dict[k] = str(my_dict[k])
        else:
            pass  # pray it can be serialized


def fix_types_iterable(iterable, output_type):
    # this sh!t is hacky as hell and will break if you use it for anything outside nnunet. Keep your hands off of this.
    out = []
    for i in iterable:
        if type(i) in (np.int64, np.int32, np.int8, np.uint8):
            out.append(int(i))
        elif isinstance(i, dict):
            recursive_fix_for_json_export(i)
            out.append(i)
        elif type(i) in (np.float32, np.float64, np.float16):
            out.append(float(i))
        elif type(i) in (np.bool_,):
            out.append(bool(i))
        elif isinstance(i, str):
            out.append(i)
        elif isinstance(i, Iterable):
            # print('recursive call on', i, type(i))
            out.append(fix_types_iterable(i, type(i)))
        else:
            out.append(i)
    return output_type(out)
