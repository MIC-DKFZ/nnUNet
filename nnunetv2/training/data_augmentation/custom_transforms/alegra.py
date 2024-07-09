from typing import Tuple, Union, Callable

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from scipy.ndimage import gaussian_filter


class InhomogeneousSliceIlluminationTransform(AbstractTransform):
    def __init__(self, num_defects, defect_width, mult_brightness_reduction_at_defect, base_p,
                 base_red: Tuple[float, float], p_per_sample=1, per_channel=True, p_per_channel=0.5, data_key='data'):
        super().__init__()
        self.num_defects = num_defects
        self.defect_width = defect_width
        self.mult_brightness_reduction_at_defect = mult_brightness_reduction_at_defect
        self.base_p = base_p
        self.base_red = base_red
        self.num_defects = num_defects
        self.p_per_sample = p_per_sample
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key

    @staticmethod
    def _sample(stuff):
        if isinstance(stuff, (float, int)):
            return stuff
        elif isinstance(stuff, (tuple, list)):
            assert len(stuff) == 2
            return np.random.uniform(*stuff)
        elif callable(stuff):
            return stuff()
        else:
            raise ValueError('hmmm')

    def _build_defects(self, num_slices):
        int_factors = np.ones(num_slices)

        # gaussian shaped ilumination changes
        num_gaussians = int(np.round(self._sample(self.num_defects)))
        for n in range(num_gaussians):
            sigma = self._sample(self.defect_width)
            pos = np.random.choice(num_slices)
            tmp = np.zeros(num_slices)
            tmp[pos] = 1
            tmp = gaussian_filter(tmp, sigma, mode='constant', truncate=3)
            tmp = tmp / tmp.max()
            strength = self._sample(self.mult_brightness_reduction_at_defect)
            int_factors *= (1 - (tmp * (1 - strength)))
        int_factors = np.clip(int_factors, 0.1, 1)
        ps = np.ones(num_slices) / num_slices
        ps += (1 - int_factors) / num_slices # probability in defect areas is twice as high as in the rest
        ps /= ps.sum()
        idx = np.random.choice(num_slices, int(np.round(self._sample(self.base_p) * num_slices)), replace=False, p=ps)
        noise = np.random.uniform(*self.base_red, size=len(idx))
        int_factors[idx] *= noise
        int_factors = np.clip(int_factors, 0.1, 2)
        return int_factors

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None
        assert len(data.shape) == 5, "this only works on 3d images, the provided tensor is 4d, so it's a 2d image (bcxy)"
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.per_channel:
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            defects = self._build_defects(data.shape[2])
                            data[b, c] *= defects[:, None, None]
                else:
                    defects = self._build_defects(data.shape[2])
                    for c in range(data.shape[1]):
                        if np.random.uniform() < self.p_per_channel:
                            data[b, c] *= defects[:, None, None]
        data_dict[self.data_key] = data
        return data_dict


class LowContrastTransform(AbstractTransform):
    def __init__(self, constrast_reduction: Union[Tuple[float, float], Callable],
                 per_channel: bool, p_per_sample: float, p_per_channel: float, data_key: str='data'):
        self.constrast_reduction = constrast_reduction
        self.per_channel = per_channel
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key

    def _get_mult_for_contrast(self):
        if isinstance(self.constrast_reduction, tuple):
            return np.random.uniform(*self.constrast_reduction)
        elif callable(self.constrast_reduction):
            return self.constrast_reduction()
        else:
            raise RuntimeError()

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            if self.p_per_sample > np.random.uniform():
                if not self.per_channel:
                    mult = self._get_mult_for_contrast()
                else:
                    mult = None
                for c in range(data.shape[1]):
                    if self.p_per_channel > np.random.uniform():
                        mn = data[b, c].mean()
                        data[b, c] -= mn
                        mult_here = self._get_mult_for_contrast() if mult is None else mult
                        data[b, c] *= mult_here
                        data[b, c] += mn
        data_dict[self.data_key] = data
        return data_dict
