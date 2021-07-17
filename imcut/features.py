#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)
import numpy as np
import scipy.ndimage


def fv_function_intensity_and_smoothing(data, voxelsize, seeds=None, unique_cls=None):
    data2 = scipy.ndimage.filters.gaussian_filter(data, sigma=5)
    # data2 = data2 - data
    arrs = [data.reshape(-1, 1), data2.reshape(-1, 1)]
    fv = np.concatenate(arrs, axis=1)
    fv2 = np.stack(arrs, -1)

    return return_fv_by_seeds(fv, seeds, unique_cls)

    # if seeds is not None:
    #     return select_from_fv_by_seeds(fv, seeds, unique_cls)
    # return fv

    # if seeds is not None:
    #     fv = []
    #     for cl in unique_cls:
    #         fv1 = data[seeds == cl].reshape(-1, 1)
    #         fv2 = data2[seeds == cl].reshape(-1, 1)
    #         fvi = np.hstack((fv1, fv2))
    #         fvi = fvi.reshape(-1, 2)
    #         fv.append(fvi)
    # else:
    #     fv1 = data.reshape(-1, 1)
    #     fv2 = data2.reshape(-1, 1)
    #     fv = np.hstack((fv1, fv2))
    #     fv = fv.reshape(-1, 2)
    #     logger.debug(str(fv[:10, :]))


def select_from_fv_by_seeds(fv, seeds, unique_cls):
    """
    Tool to make simple feature functions take features from feature array by seeds.
    :param fv: ndarray with lineariezed feature. It's shape is MxN, where M is number of image pixels and N is number
    of features
    :param seeds: ndarray with seeds. Does not to be linear.
    :param unique_cls: number of used seeds clases. Like [1, 2]
    :return: fv_selection, seeds_selection - selection from feature vector and selection from seeds
    """
    logger.debug("seeds" + str(seeds))
    # fvlin = fv.reshape(-1, int(fv.size/seeds.size))
    expected_shape = [seeds.size, int(fv.size / seeds.size)]
    if fv.shape[0] != expected_shape[0] or fv.shape[1] != expected_shape[1]:
        raise AssertionError("Wrong shape of input feature vector array fv")
    # sd = seeds.reshape(-1, 1)
    selection = np.in1d(seeds, unique_cls)
    fv_selection = fv[selection]
    seeds_selection = seeds.flatten()[selection]
    # sd = sd[]
    return fv_selection, seeds_selection


def return_fv_by_seeds(fv, seeds=None, unique_cls=None):
    """
    Return features selected by seeds and unique_cls or selection from features and corresponding seed classes.

    :param fv: ndarray with lineariezed feature. It's shape is MxN, where M is number of image pixels and N is number
    of features
    :param seeds: ndarray with seeds. Does not to be linear.
    :param unique_cls: number of used seeds clases. Like [1, 2]
    :return: fv, sd - selection from feature vector and selection from seeds or just fv for whole image
    """
    if seeds is not None:
        if unique_cls is not None:
            return select_from_fv_by_seeds(fv, seeds, unique_cls)
        else:
            raise AssertionError(
                "Input unique_cls has to be not None if seeds is not None."
            )
    else:
        return fv
