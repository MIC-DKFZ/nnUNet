from typing import Tuple, Union, List

from batchgenerators.transforms.abstract_transforms import AbstractTransform


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self, apply_to_keys: Union[List[str], Tuple[str]] = ('data', 'seg')):
        """
        Transforms a 5D array (b, c, x, y, z) to a 4D array (b, c * x, y, z) by overloading the color channel
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict):
        for k in self.apply_to_keys:
            shp = data_dict[k].shape
            assert len(shp) == 5, 'This transform only works on 3D data, so expects 5D tensor (b, c, x, y, z) as input.'
            data_dict[k] = data_dict[k].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
            shape_key = f'orig_shape_{k}'
            assert shape_key not in data_dict.keys(), f'Convert3DTo2DTransform needs to store the original shape. ' \
                                                      f'It does that using the {shape_key} key. That key is ' \
                                                      f'already taken. Bummer.'
            data_dict[shape_key] = shp
        return data_dict


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self, apply_to_keys: Union[List[str], Tuple[str]] = ('data', 'seg')):
        """
        Reverts Convert3DTo2DTransform by transforming a 4D array (b, c * x, y, z) back to 5D  (b, c, x, y, z)
        """
        self.apply_to_keys = apply_to_keys

    def __call__(self, **data_dict):
        for k in self.apply_to_keys:
            shape_key = f'orig_shape_{k}'
            assert shape_key in data_dict.keys(), f'Did not find key {shape_key} in data_dict. Shitty. ' \
                                                  f'Convert2DTo3DTransform only works in tandem with ' \
                                                  f'Convert3DTo2DTransform and you probably forgot to add ' \
                                                  f'Convert3DTo2DTransform to your pipeline. (Convert3DTo2DTransform ' \
                                                  f'is where the missing key is generated)'
            original_shape = data_dict[shape_key]
            current_shape = data_dict[k].shape
            data_dict[k] = data_dict[k].reshape((original_shape[0], original_shape[1], original_shape[2],
                                                 current_shape[-2], current_shape[-1]))
        return data_dict
