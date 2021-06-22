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

import random
import numpy as np
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares
from builtins import range
from scipy.ndimage import gaussian_filter


def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
    return data_sample


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


def augment_gaussian_blur(data_sample, sigma_range, per_channel=True, p_per_channel=1):
    if not per_channel:
        sigma = get_range_val(sigma_range)
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range)
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


def augment_blank_square_noise(data_sample, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False,
                               square_pos=None):
    # rnd_n_val = get_range_val(noise_val)
    rnd_square_size = get_range_val(square_size)
    rnd_n_squares = get_range_val(n_squares)

    data_sample = mask_random_squares(data_sample, square_size=rnd_square_size, n_squares=rnd_n_squares,
                                           n_val=noise_val, channel_wise_n_val=channel_wise_n_val,
                                           square_pos=square_pos)
    return data_sample


