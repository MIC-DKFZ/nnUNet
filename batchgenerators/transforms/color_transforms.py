# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC

import numpy as np
from scipy.ndimage import gaussian_filter

from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
    augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from typing import Union, Tuple, Callable, List
import scipy.stats as st


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 data_key: str = "data",
                 p_per_sample: float = 1,
                 p_per_channel: float = 1):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
                                                               contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel,
                                                               p_per_channel=self.p_per_channel)
        return data_dict


class NormalizeTransform(AbstractTransform):
    def __init__(self, means, stds, data_key='data'):
        self.data_key = data_key
        self.stds = stds
        self.means = means

    def __call__(self, **data_dict):
        for c in range(data_dict[self.data_key].shape[1]):
            data_dict[self.data_key][:, c] -= self.means[c]
            data_dict[self.data_key][:, c] /= self.stds[c]
        return data_dict


class BrightnessTransform(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True, data_key="data", p_per_sample=1, p_per_channel=1):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        data_dict[self.data_key] = data
        return data_dict


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
                                                                                self.multiplier_range,
                                                                                self.per_channel)
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict


class IlluminationTransform(AbstractTransform):
    """Do not use this for now"""

    def __init__(self, white_rgb, data_key="data"):
        self.data_key = data_key
        self.white_rgb = white_rgb

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_illumination(data_dict[self.data_key], self.white_rgb)
        return data_dict


class FancyColorTransform(AbstractTransform):
    """Do not use this for now"""

    def __init__(self, U, s, sigma=0.2, data_key="data"):
        self.data_key = data_key
        self.s = s
        self.U = U
        self.sigma = sigma

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_PCA_shift(data_dict[self.data_key], self.U, self.s, self.sigma)
        return data_dict


class ClipValueRange(AbstractTransform):
    def __init__(self, min=None, max=None, data_key="data"):
        """
        Clips the value range of data to [min, max]
        :param min:
        :param max:
        :param data_key:
        """
        self.data_key = data_key
        self.min = min
        self.max = max

    def __call__(self, **data_dict):
        data_dict[self.data_key] = np.clip(data_dict[self.data_key], self.min, self.max)
        return data_dict


class LocalGaussianSomethingTransform(ABC):
    def __init__(self,
                 scale: Union[Tuple[float, float], float, Callable[[Union[Tuple[int, ...], List[int]], int], float]],
                 loc: Union[Tuple[float, float], Callable[[Union[Tuple[int, ...], List[int]], int], float]] = (-1, 2),
                 ):
        self.loc = loc
        self.scale = scale

    def _get_scale(self, image_shape, dimension):
        if isinstance(self.scale, float):
            return self.scale
        elif isinstance(self.scale, (list, tuple)):
            assert len(self.scale) == 2
            return np.random.uniform(*self.scale)
        elif callable(self.scale):
            return self.scale(image_shape, dimension)

    def _get_loc(self, image_shape, dimension):
        if isinstance(self.loc, float):
            return self.loc
        elif isinstance(self.loc, (list, tuple)):
            assert len(self.loc) == 2
            return np.random.uniform(*self.loc)
        elif callable(self.loc):
            return self.loc(image_shape, dimension)
        else:
            raise RuntimeError()

    def _generate_kernel(self, img_shp: Tuple[int, ...]) -> np.ndarray:
        assert len(img_shp) <= 3
        kernels = []
        for d in range(len(img_shp)):
            image_size_here = img_shp[d]
            loc = self._get_loc(img_shp, d)
            scale = self._get_scale(img_shp, d)

            loc_rescaled = loc * image_size_here
            x = np.arange(-0.5, image_size_here + 0.5)
            kernels.append(np.diff(st.norm.cdf(x, loc=loc_rescaled, scale=scale)))

        kernel_2d = kernels[0][:, None].dot(kernels[1][None])
        if len(kernels) > 2:
            # trial and error got me here lol
            kernel = kernel_2d[:, :, None].dot(kernels[2][None])
        else:
            kernel = kernel_2d
        return kernel


class BrightnessGradientAdditiveTransform(LocalGaussianSomethingTransform):
    def __init__(self,
                 scale: Union[Tuple[float, float], float, Callable[[Union[Tuple[int, ...], List[int]], int], float]],
                 loc: Union[Tuple[float, float], Callable[[Union[Tuple[int, ...], List[int]], int], float]] = (-1, 2),
                 max_strength: Union[float, Tuple[float, float], Callable[[np.ndarray, np.ndarray], float]] = 1.,
                 same_for_all_channels: bool = True,
                 mean_centered: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 clip_intensities: bool = False,
                 data_key: str = "data"):
        """
        Applies an additive intensity gradient to the image. The intensity gradient is zero-centered (sum(add) = 0;
        will not shift the global mean of the image. Some pixels will be brighter, some darker after application)

        The gradient is implemented by placing a Gaussian distribution with sigma=scale somewhere in the image. The
        location of the kernel is selected independently for each image dimension. The location is encoded in % of the
        image size. The default value of (-1, 2) means that the location will be sampled uniformly from
        (-image.shape[i], 2* image.shape[i]). It is important to allow the center of the kernel to be outside of the image.

        IMPORTANT: Try this with different parametrizations and visualize the outcome to get a better feeling for how
        to use this!

        :param scale: scale of the gradient. Large values recommended!
            float: fixed value
            (float, float): will be sampled independently for each dimension from the interval [scale[0], scale[1]]
            callable: you get all the freedom you want. Will be called as scale(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the scale for. Must return scalar (float).
        :param loc:
            (float, float): sample location uniformly from interval [scale[0], scale[1]] (see main description)
            callable: you get all the freedom you want. Will be called as loc(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the location for. Must return a scalar value denoting a relative
            position along axis dimension (0 for index 0, 1 for image.shape[dimension]. Values beyond 0 and 1 are
            possible and even recommended)
        :param max_strength: scaling of the intensity gradient. Determines what max(abs(add_gauss)) is going to be
            float: fixed value
            (float, float): sampled from [max_strength[0], max_strength[1]]
            callable: you decide. Will be called as max_strength(image, gauss_add). Do not modify gauss_add.
            Must return a scalar.
        :param same_for_all_channels: If True, then the same gradient will be applied to all selected color
        channels of a sample (see p_per_channel). If False, each selected channel obtains its own random gradient.
        :param mean_centered: if True, the brightness addition will be done such that the mean intensity of the image
        does not change. So if a bright spot is added, other parts of the image will have something subtracted to keep
        the mean intensity the same as it was before
        :param p_per_sample:
        :param p_per_channel:
        :param clip_intensities:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.max_strength = max_strength
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.same_for_all_channels = same_for_all_channels
        self.mean_centered = mean_centered
        self.clip_intensities = clip_intensities

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)
                    if self.mean_centered:
                        # first center the mean of the kernel
                        kernel -= kernel.mean()
                    mx = max(np.max(np.abs(kernel)), 1e-8)
                    if not callable(self.max_strength):
                        strength = self._get_max_strength(None, None)
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            # now rescale so that the maximum value of the kernel is max_strength
                            strength = self._get_max_strength(data[bi, ci], kernel) if callable(
                                self.max_strength) else strength
                            kernel_scaled = np.copy(kernel) / mx * strength
                            data[bi, ci] += kernel_scaled
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            if self.mean_centered:
                                kernel -= kernel.mean()
                            mx = max(np.max(np.abs(kernel)), 1e-8)
                            strength = self._get_max_strength(data[bi, ci], kernel)
                            kernel = kernel / mx * strength
                            data[bi, ci] += kernel
        return data_dict

    def _get_max_strength(self, image, add_gauss):
        if isinstance(self.max_strength, (int, float)):
            return self.max_strength
        elif isinstance(self.max_strength, (list, tuple)):
            assert len(self.max_strength) == 2
            return np.random.uniform(*self.max_strength)
        elif callable(self.max_strength):
            return self.max_strength(image, add_gauss)
        else:
            raise RuntimeError()


class LocalGammaTransform(LocalGaussianSomethingTransform):
    def __init__(self,
                 scale: Union[Tuple[float, float], float, Callable[[Union[Tuple[int, ...], List[int]], int], float]],
                 loc: Union[Tuple[float, float], Callable[[Union[Tuple[int, ...], List[int]], int], float]] = (-1, 2),
                 gamma: Union[float, Tuple[float, float], Callable[[], float]] = (0.5, 1),
                 same_for_all_channels: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key: str = "data"):
        """
        This transform is weird and definitely experimental.

        Applies gamma correction to the image. The gamma value varies locally using a gaussian kernel

        The local gamma values are implemented by placing a Gaussian distribution with sigma=scale somewhere in
        (or close to) the image. The location of the kernel is selected independently for each image dimension.
        The location is encoded in % of the image size. The default value of (-1, 2) means that the location will be
        sampled uniformly from (-image.shape[i], 2* image.shape[i]). It is important to allow the center of the kernel
        to be outside of the image.

        IMPORTANT: Try this with different parametrizations and visualize the outcome to get a better feeling for how
        to use this!

        :param scale: scale of the gradient. Large values recommended!
            float: fixed value
            (float, float): will be sampled independently for each dimension from the interval [scale[0], scale[1]]
            callable: you get all the freedom you want. Will be called as scale(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the scale for. Must return scalar (float).
        :param loc:
            (float, float): sample location uniformly from interval [scale[0], scale[1]] (see main description)
            callable: you get all the freedom you want. Will be called as loc(image.shape, dimension) where dimension
            is the index in image.shape we are requesting the location for. Must return a scalar value denoting a relative
            position along axis dimension (0 for index 0, 1 for image.shape[dimension]. Values beyond 0 and 1 are
            possible and even recommended)
        :param gamma: gamma value at the peak of the gaussian distribution.
            Recommended: lambda: np.random.uniform(0.01, 1) if np.random.uniform() < 1 else np.random.uniform(1, 3)
            No, this is not a joke. Deal with it.
        :param same_for_all_channels: If True, then the same gradient will be applied to all selected color
        channels of a sample (see p_per_channel). If False, each selected channel obtains its own random gradient.
        :param allow_kernel_inversion:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.gamma = gamma
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.same_for_all_channels = same_for_all_channels

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)

                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            data[bi, ci] = self._apply_gamma_gradient(data[bi, ci], kernel)
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            data[bi, ci] = self._apply_gamma_gradient(data[bi, ci], kernel)
        return data_dict

    def _get_gamma(self):
        if isinstance(self.gamma, (int, float)):
            return self.gamma
        elif isinstance(self.gamma, (list, tuple)):
            assert len(self.gamma) == 2
            return np.random.uniform(*self.gamma)
        elif callable(self.gamma):
            return self.gamma()
        else:
            raise RuntimeError()

    def _apply_gamma_gradient(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # copy kernel so that we don't modify it out of scope
        kernel = np.copy(kernel)

        # store keep original image range
        mn, mx = img.min(), img.max()

        # rescale tp [0, 1]
        img = (img - mn) / (max(mx - mn, 1e-8))

        inside_gamma = self._get_gamma()
        outside_gamma = 1

        # prepare kernel by rescaling it to gamma_range
        k_min, k_max = kernel.min(), kernel.max()
        kernel = (kernel - k_min) / (max(k_max - k_min, 1e-8))  # [0, 1]
        kernel *= (inside_gamma - outside_gamma)
        kernel += outside_gamma

        # corrected = image ** gamma
        img = np.power(img, kernel)

        # restore original range
        img = img * (mx - mn) + mn
        return img


class LocalSmoothingTransform(LocalGaussianSomethingTransform):
    def __init__(self,
                 scale: Union[Tuple[float, float], float, Callable[[Union[Tuple[int, ...], List[int]], int], float]],
                 loc: Union[Tuple[float, float], Callable[[Union[Tuple[int, ...], List[int]], int], float]] = (-1, 2),
                 smoothing_strength: Union[float, Tuple[float, float], Callable[[], float]] = (0.5, 1),
                 kernel_size: Union[float, Tuple[float, float], Callable[[], float]] = (0.5, 1),
                 same_for_all_channels: bool = True,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 data_key: str = "data"):
        """
        Creates a local blurring of the image. This is achieved by creating ablussed copy of the image and then
        linearly interpolating between the original and smoothed images:
            result = image_orig * (1 - smoothing_strength) + smoothed_image * smoothing_strength
        The interpolation only happens where the local gaussian is placed (defined by scale and loc)
        strength of smoothing is determined by kernel_size in combination with smoothing_strength. You can set
        smoothing_strength=1 for simplicity
        :param scale:
        :param loc:
        :param smoothing_strength:
        :param kernel_size:
        :param same_for_all_channels:
        :param p_per_sample:
        :param p_per_channel:
        :param data_key:
        """
        super().__init__(scale, loc)
        self.smoothing_strength = smoothing_strength
        self.same_for_all_channels = same_for_all_channels
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.kernel_size = kernel_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        assert data is not None, "Could not find data key '%s'" % self.data_key
        b, c, *img_shape = data.shape
        for bi in range(b):
            if np.random.uniform() < self.p_per_sample:
                if self.same_for_all_channels:
                    kernel = self._generate_kernel(img_shape)

                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
                else:
                    for ci in range(c):
                        if np.random.uniform() < self.p_per_channel:
                            kernel = self._generate_kernel(img_shape)
                            data[bi, ci] = self._apply_local_smoothing(data[bi, ci], kernel)
        return data_dict

    def _get_smoothing(self):
        if isinstance(self.smoothing_strength, (int, float)):
            return self.smoothing_strength
        elif isinstance(self.smoothing_strength, (list, tuple)):
            assert len(self.smoothing_strength) == 2
            return np.random.uniform(*self.smoothing_strength)
        elif callable(self.smoothing_strength):
            return self.smoothing_strength()
        else:
            raise RuntimeError()

    def _get_kernel_size(self):
        if isinstance(self.kernel_size, (int, float)):
            return self.kernel_size
        elif isinstance(self.kernel_size, (list, tuple)):
            assert len(self.kernel_size) == 2
            return np.random.uniform(*self.kernel_size)
        elif callable(self.kernel_size):
            return self.kernel_size()
        else:
            raise RuntimeError()

    def _apply_local_smoothing(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # copy kernel so that we don't modify it out of scope
        kernel = np.copy(kernel)

        smoothing = self._get_smoothing()

        # prepare kernel by rescaling it to gamma_range
        k_min, k_max = kernel.min(), kernel.max()
        kernel = (kernel - k_min) / (max(k_max - k_min, 1e-8))  # [0, 1]
        kernel *= smoothing

        kernel_size = self._get_kernel_size()
        img_smoothed = gaussian_filter(img, kernel_size)

        return img * (1 - kernel) + img_smoothed * kernel

"""class LocalContastTransform(LocalGammaTransform):
    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        ret = super().__call__(**{'data': np.copy(data)})['data']
        filter_orig = np.zeros_like(data)
        filter_aug = np.zeros_like(data)
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                filter_orig[b, c] = gaussian_filter(np.copy(data[b, c]), sigma=5)
                filter_aug[b, c] = gaussian_filter(np.copy(ret[b, c]), sigma=5)
        intens_corrected = np.clip(ret * (filter_orig / filter_aug), data.min(), data.max())
        #import IPython;IPython.embed()
        data_dict['data'] = intens_corrected
        return data_dict"""


if __name__ == '__main__':
    from copy import deepcopy
    from skimage.data import camera

    # just some playing around with BrightnessGradientAdditiveTransform
    data = {'data': np.vstack((camera()[None], camera()[None], camera()[None]))[None].astype(np.float32)}
    tr = LocalSmoothingTransform(
        lambda x, y: np.random.uniform(x[y] // 6, x[y] // 2),
        (0, 1),
        1,
        13,
        same_for_all_channels=False
    )
    transformed = tr(**deepcopy(data))['data']
    from batchviewer import view_batch
    data['data'][0][:, 0:2, 0] = np.array((0, 255))
    transformed[0][:, 0:2, 0] = np.array((0, 255))
    diff = [i - j for i, j in zip(data['data'][0], transformed[0])]
    [print(i[10,10]) for i in diff]
    view_batch(*data['data'][0], *transformed[0], *[i - j for i, j in zip(data['data'][0], transformed[0])])
