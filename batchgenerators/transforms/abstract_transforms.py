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


import abc
from warnings import warn

import numpy as np


class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class RndTransform(AbstractTransform):
    """Applies a transformation with a specified probability

    Args:
        transform: The transformation (or composed transformation)

        prob: The probability with which to apply it

        alternative_transform: Will be applied if transform is not called. If transform alters for example the
        spatial dimension of the data, you need to compensate that with calling a dummy transformation that alters the
        spatial dimension in a similar way. We included this functionality because of SpatialTransform which has the
        ability to do cropping. If we want to not apply the spatial transformation we will still need to crop and
        therefore set the alternative_transform to an instance of RandomCropTransform of CenterCropTransform
    """

    def __init__(self, transform, prob=0.5, alternative_transform=None):
        warn("This is deprecated. All applicable transfroms now have a p_per_sample argument which allows "
             "batchgenerators to do or not do an augmentation on a per-sample basis instead of the entire batch",
             DeprecationWarning)
        self.alternative_transform = alternative_transform
        self.transform = transform
        self.prob = prob

    def __call__(self, **data_dict):
        rnd_val = np.random.uniform()

        if rnd_val < self.prob:
            return self.transform(**data_dict)
        else:
            if self.alternative_transform is not None:
                return self.alternative_transform(**data_dict)
            else:
                return data_dict


class Compose(AbstractTransform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"
