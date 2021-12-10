#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

# import os.path as op

logger = logging.getLogger(__name__)
import numpy as np
import scipy
import scipy.ndimage


def resize_to_shape(data, shape, zoom=None, mode="nearest", order=0):
    """
    Function resize input data to specific shape.
    :param data: input 3d array-like data
    :param shape: shape of output data
    :param zoom: zoom is used for back compatibility
    :mode: default is 'nearest'
    """
    # @TODO remove old code in except part
    # TODO use function from library in future

    try:
        # rint 'pred vyjimkou'
        # aise Exception ('test without skimage')
        # rint 'za vyjimkou'
        import skimage
        import skimage.transform

        # Now we need reshape  seeds and segmentation to original size

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", ".*'constant', will be changed to.*")
        segm_orig_scale = skimage.transform.resize(
            data, shape, order=0, preserve_range=True, mode="reflect"
        )

        segmentation = segm_orig_scale
        logger.debug("resize to orig with skimage")
    except:
        if zoom is None:
            zoom = shape / np.asarray(data.shape).astype(np.double)
        segmentation = resize_to_shape_with_zoom(
            data, zoom=zoom, mode=mode, order=order
        )

    return segmentation


def resize_to_shape_with_zoom(data, shape, zoom, mode="nearest", order=0):
    import scipy
    import scipy.ndimage

    dtype = data.dtype

    segm_orig_scale = scipy.ndimage.zoom(
        data, 1.0 / zoom, mode=mode, order=order
    ).astype(dtype)
    logger.debug("resize to orig with scipy.ndimage")

    # @TODO odstranit hack pro oříznutí na stejnou velikost
    # v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
    # tam je bohužel patrně bug
    # rint 'd3d ', self.data3d.shape
    # rint 's orig scale shape ', segm_orig_scale.shape
    shp = [
        np.min([segm_orig_scale.shape[0], shape[0]]),
        np.min([segm_orig_scale.shape[1], shape[1]]),
        np.min([segm_orig_scale.shape[2], shape[2]]),
    ]
    # elf.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]
    # mport ipdb; ipdb.set_trace() # BREAKPOINT

    segmentation = np.zeros(shape, dtype=dtype)
    segmentation[0 : shp[0], 0 : shp[1], 0 : shp[2]] = segm_orig_scale[
        0 : shp[0], 0 : shp[1], 0 : shp[2]
    ]

    del segm_orig_scale
    return segmentation


def seed_zoom(seeds, zoom):
    """
    Smart zoom for sparse matrix. If there is resize to bigger resolution
    thin line of label could be lost. This function prefers labels larger
    then zero. If there is only one small voxel in larger volume with zeros
    it is selected.
    """
    # import scipy
    # loseeds=seeds
    labels = np.unique(seeds)
    # remove first label - 0
    labels = np.delete(labels, 0)
    # @TODO smart interpolation for seeds in one block
    #        loseeds = scipy.ndimage.interpolation.zoom(
    #            seeds, zoom, order=0)
    loshape = np.ceil(np.array(seeds.shape) * 1.0 / zoom).astype(np.int)
    loseeds = np.zeros(loshape, dtype=np.int8)
    loseeds = loseeds.astype(np.int8)
    for label in labels:
        a, b, c = np.where(seeds == label)
        loa = np.round(a // zoom)
        lob = np.round(b // zoom)
        loc = np.round(c // zoom)
        # loseeds = np.zeros(loshape)

        loseeds[loa, lob, loc] += label
        # this is to detect conflict seeds
        loseeds[loseeds > label] = 100

    # remove conflict seeds
    loseeds[loseeds > 99] = 0

    # import py3DSeedEditor
    # ped = py3DSeedEditor.py3DSeedEditor(loseeds)
    # ped.show()

    return loseeds


def zoom_to_shape(data, shape, dtype=None):
    """
    Zoom data to specific shape.
    """
    import scipy
    import scipy.ndimage

    zoomd = np.array(shape) / np.array(data.shape, dtype=np.double)
    import warnings

    datares = scipy.ndimage.interpolation.zoom(data, zoomd, order=0, mode="reflect")

    if datares.shape != shape:
        logger.warning("Zoom with different output shape")
    dataout = np.zeros(shape, dtype=dtype)
    shpmin = np.minimum(dataout.shape, shape)

    dataout[: shpmin[0], : shpmin[1], : shpmin[2]] = datares[
        : shpmin[0], : shpmin[1], : shpmin[2]
    ]
    return datares


def select_objects_by_seeds(
    binar_data, seeds, ignore_background_seeds=True, background_label=0
):

    labeled_data, length = scipy.ndimage.label(binar_data)
    selected_labels = list(np.unique(labeled_data[seeds > 0]))
    # selected_labels.pop(0)
    # pop the background label
    output = np.zeros_like(binar_data)
    for label in selected_labels:
        selection = labeled_data == label
        # copy from input image to output. If there will be seeds in background, the 0 is copied
        if ignore_background_seeds and (binar_data[selection][0] == background_label):
            pass
        else:
            # output[selection] = binar_data[selection]
            output[selection] = 1
    # import sed3
    # ed =sed3.sed3(labeled_data, contour=output, seeds=seeds)
    # ed.show()
    return output


# def getPriorityObjects(*args, **kwargs):
#     logger.warning("Function getPriorityObjects has been renamed. Use get_priority_objects().")
#     DeprecationWarning("Function getPriorityObjects has been renamed. Use get_priority_objects().")
#     return get_priority_objects(*args, **kwargs)
#
# def get_priority_objects(data, nObj=1, seeds=None, debug=False):
#     """
#     Get N biggest objects from the selection or the object with seed.
#
#     :param data:  labeled ndarray
#     :param nObj:  number of objects
#     :param seeds: ndarray. Objects on non zero positions are returned
#     :param debug: bool.
#     :return: binar image with selected objects
#     """
#
#     # Oznaceni dat.
#     # labels - oznacena data.
#     # length - pocet rozdilnych oznaceni.
#     dataLabels, length = scipy.ndimage.label(data)
#
#     logger.info('Olabelovano oblasti: ' + str(length))
#
#     if debug:
#         logger.debug('data labels: ' + str(dataLabels))
#
#     # Uzivatel si nevybral specificke objekty.
#     if (seeds == None):
#
#         logger.info('Vraceni bez seedu')
#         logger.debug('Objekty: ' + str(nObj))
#
#         # Zjisteni nejvetsich objektu.
#         arrayLabelsSum, arrayLabels = areaIndexes(dataLabels, length)
#         # Serazeni labelu podle velikosti oznacenych dat (prvku / ploch).
#         arrayLabelsSum, arrayLabels = selectSort(arrayLabelsSum, arrayLabels)
#
#         returning = None
#         label = 0
#         stop = nObj - 1
#
#         # Budeme postupne prochazet arrayLabels a postupne pridavat jednu
#         # oblast za druhou (od te nejvetsi - mimo nuloveho pozadi) dokud
#         # nebudeme mit dany pocet objektu (nObj).
#         while label <= stop:
#
#             if label >= len(arrayLabels):
#                 break
#
#             if arrayLabels[label] != 0:
#                 if returning == None:
#                     # "Prvni" iterace
#                     returning = data * (dataLabels == arrayLabels[label])
#                 else:
#                     # Jakakoli dalsi iterace
#                     returning = returning + data * \
#                                 (dataLabels == arrayLabels[label])
#             else:
#                 # Musime prodlouzit hledany interval, protoze jsme narazili na
#                 # nulove pozadi.
#                 stop = stop + 1
#
#             label = label + 1
#
#             if debug:
#                 logger.debug(str(label - 1) + ': ' + str(returning))
#
#         if returning == None:
#             logger.info(
#                 'Zadna validni olabelovana data! (DEBUG: returning == None)')
#
#         return returning
#
#     # Uzivatel si vybral specificke objekty (seeds != None).
#     else:
#
#         logger.info('Vraceni se seedy')
#
#         # Zalozeni pole pro ulozeni seedu
#         arrSeed = []
#         # Zjisteni poctu seedu.
#         stop = seeds[0].size
#         tmpSeed = 0
#         dim = np.ndim(dataLabels)
#         for index in range(0, stop):
#             # Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
#             if dim == 3:
#                 # 3D data.
#                 tmpSeed = dataLabels[
#                     seeds[0][index], seeds[1][index], seeds[2][index]]
#             elif dim == 2:
#                 # 2D data.
#                 tmpSeed = dataLabels[seeds[0][index], seeds[1][index]]
#
#             # Tady opet pocitam s tim, ze oznaceni nulou pripada cerne oblasti
#             # (pozadi).
#             if tmpSeed != 0:
#                 # Pokud se nejedna o pozadi (cernou oblast), tak se novy seed
#                 # ulozi do pole "arrSeed"
#                 arrSeed.append(tmpSeed)
#
#         # Pokud existuji vhodne labely, vytvori se nova data k vraceni.
#         # Pokud ne, vrati se "None" typ. { Deprecated: Pokud ne, vrati se cela
#         # nafiltrovana data, ktera do funkce prisla (nedojde k vraceni
#         # specifickych objektu). }
#         if len(arrSeed) > 0:
#
#             # Zbaveni se duplikatu.
#             arrSeed = list(set(arrSeed))
#             if debug:
#                 logger.debug('seed list:' + str(arrSeed))
#
#             logger.info(
#                 'Ruznych prioritnich objektu k vraceni: ' +
#                 str(len(arrSeed))
#             )
#
#             # Vytvoreni vystupu - postupne pricitani dat prislunych specif.
#             # labelu.
#             returning = None
#             for index in range(0, len(arrSeed)):
#
#                 if returning == None:
#                     returning = data * (dataLabels == arrSeed[index])
#                 else:
#                     returning = returning + data * \
#                                 (dataLabels == arrSeed[index])
#
#                 if debug:
#                     logger.debug((str(index)) + ':' + str(returning))
#
#             return returning
#
#         else:
#
#             logger.info(
#                 'Zadna validni data k vraceni - zadne prioritni objekty ' +
#                 'nenalezeny (DEBUG: function getPriorityObjects:' +
#                 str(len(arrSeed) == 0))
#             return None
#
# def areaIndexes(labels, num):
#     """
#
#     Zjisti cetnosti jednotlivych oznacenych ploch (labeled areas)
#         input:
#             labels - data s aplikovanymi oznacenimi
#             num - pocet pouzitych oznaceni
#
#         returns:
#             dve pole - prvni sumy, druhe indexy
#
#     """
#
#     arrayLabelsSum = []
#     arrayLabels = []
#     for index in range(0, num + 1):
#         arrayLabels.append(index)
#         sumOfLabel = numpy.sum(labels == index)
#         arrayLabelsSum.append(sumOfLabel)
#
#     return arrayLabelsSum, arrayLabels
#
#
# def selectSort(list1, list2):
#     """
#     Razeni 2 poli najednou (list) pomoci metody select sort
#         input:
#             list1 - prvni pole (hlavni pole pro razeni)
#             list2 - druhe pole (vedlejsi pole) (kopirujici pozice pro razeni
#                 podle hlavniho pole list1)
#
#         returns:
#             dve serazena pole - hodnoty se ridi podle prvniho pole, druhe
#                 "kopiruje" razeni
#     """
#
#     length = len(list1)
#     for index in range(0, length):
#         min = index
#         for index2 in range(index + 1, length):
#             if list1[index2] > list1[min]:
#                 min = index2
#         # Prohozeni hodnot hlavniho pole
#         list1[index], list1[min] = list1[min], list1[index]
#         # Prohozeni hodnot vedlejsiho pole
#         list2[index], list2[min] = list2[min], list2[index]
#
#     return list1, list2


def crop(data, crinfo):
    """
    Crop the data.

    crop(data, crinfo)

    :param crinfo: min and max for each axis - [[minX, maxX], [minY, maxY], [minZ, maxZ]]

    """
    crinfo = fix_crinfo(crinfo)
    return data[
        __int_or_none(crinfo[0][0]) : __int_or_none(crinfo[0][1]),
        __int_or_none(crinfo[1][0]) : __int_or_none(crinfo[1][1]),
        __int_or_none(crinfo[2][0]) : __int_or_none(crinfo[2][1]),
    ]


def __int_or_none(number):
    if number is not None:
        number = int(number)
    return number


def combinecrinfo(crinfo1, crinfo2):
    """
    Combine two crinfos. First used is crinfo1, second used is crinfo2.
    """
    crinfo1 = fix_crinfo(crinfo1)
    crinfo2 = fix_crinfo(crinfo2)

    crinfo = [
        [crinfo1[0][0] + crinfo2[0][0], crinfo1[0][0] + crinfo2[0][1]],
        [crinfo1[1][0] + crinfo2[1][0], crinfo1[1][0] + crinfo2[1][1]],
        [crinfo1[2][0] + crinfo2[2][0], crinfo1[2][0] + crinfo2[2][1]],
    ]

    return crinfo


def crinfo_from_specific_data(data, margin=0):
    """
    Create crinfo of minimum orthogonal nonzero block in input data.

    :param data: input data
    :param margin: add margin to minimum block
    :return:
    """
    # hledáme automatický ořez, nonzero dá indexy
    logger.debug("crinfo")
    logger.debug(str(margin))
    nzi = np.nonzero(data)
    logger.debug(str(nzi))

    if np.isscalar(margin):
        margin = [margin] * 3

    x1 = np.min(nzi[0]) - margin[0]
    x2 = np.max(nzi[0]) + margin[0] + 1
    y1 = np.min(nzi[1]) - margin[0]
    y2 = np.max(nzi[1]) + margin[0] + 1
    z1 = np.min(nzi[2]) - margin[0]
    z2 = np.max(nzi[2]) + margin[0] + 1

    # ošetření mezí polí
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if z1 < 0:
        z1 = 0

    if x2 > data.shape[0]:
        x2 = data.shape[0] - 1
    if y2 > data.shape[1]:
        y2 = data.shape[1] - 1
    if z2 > data.shape[2]:
        z2 = data.shape[2] - 1

    # ořez
    crinfo = [[x1, x2], [y1, y2], [z1, z2]]
    return crinfo


def uncrop(data, crinfo, orig_shape, resize=False, outside_mode="constant", cval=0):
    """
    Put some boundary to input image.


    :param data: input data
    :param crinfo: array with minimum and maximum index along each axis
        [[minX, maxX],[minY, maxY],[minZ, maxZ]]. If crinfo is None, the whole input image is placed into [0, 0, 0].
        If crinfo is just series of three numbers, it is used as an initial point for input image placement.
    :param orig_shape: shape of uncropped image
    :param resize: True or False (default). Usefull if the data.shape does not fit to crinfo shape.
    :param outside_mode: 'constant', 'nearest'
    :return:
    """

    if crinfo is None:
        crinfo = list(zip([0] * data.ndim, orig_shape))
    elif np.asarray(crinfo).size == data.ndim:
        crinfo = list(zip(crinfo, np.asarray(crinfo) + data.shape))

    crinfo = fix_crinfo(crinfo)
    data_out = np.ones(orig_shape, dtype=data.dtype) * cval

    # print 'uncrop ', crinfo
    # print orig_shape
    # print data.shape
    if resize:
        data = resize_to_shape(data, crinfo[:, 1] - crinfo[:, 0])

    startx = np.round(crinfo[0][0]).astype(int)
    starty = np.round(crinfo[1][0]).astype(int)
    startz = np.round(crinfo[2][0]).astype(int)

    data_out[
        # np.round(crinfo[0][0]).astype(int):np.round(crinfo[0][1]).astype(int)+1,
        # np.round(crinfo[1][0]).astype(int):np.round(crinfo[1][1]).astype(int)+1,
        # np.round(crinfo[2][0]).astype(int):np.round(crinfo[2][1]).astype(int)+1
        startx : startx + data.shape[0],
        starty : starty + data.shape[1],
        startz : startz + data.shape[2],
    ] = data

    if outside_mode == "nearest":
        # for ax in range(data.ndims):
        # ax = 0

        # copy border slice to pixels out of boundary - the higher part
        for ax in range(data.ndim):
            # the part under the crop
            start = np.round(crinfo[ax][0]).astype(int)
            slices = [slice(None), slice(None), slice(None)]
            slices[ax] = start
            repeated_slice = np.expand_dims(data_out[tuple(slices)], ax)
            append_sz = start
            if append_sz > 0:
                tile0 = np.repeat(repeated_slice, append_sz, axis=ax)
                slices = [slice(None), slice(None), slice(None)]
                slices[ax] = slice(None, start)
                # data_out[start + data.shape[ax] : , :, :] = tile0
                data_out[slices] = tile0
                # plt.imshow(np.squeeze(repeated_slice))
                # plt.show()

            # the part over the crop
            start = np.round(crinfo[ax][0]).astype(int)
            slices = [slice(None), slice(None), slice(None)]
            slices[ax] = start + data.shape[ax] - 1
            repeated_slice = np.expand_dims(data_out[tuple(slices)], ax)
            append_sz = data_out.shape[ax] - (start + data.shape[ax])
            if append_sz > 0:
                tile0 = np.repeat(repeated_slice, append_sz, axis=ax)
                slices = [slice(None), slice(None), slice(None)]
                slices[ax] = slice(start + data.shape[ax], None)
                # data_out[start + data.shape[ax] : , :, :] = tile0
                data_out[tuple(slices)] = tile0
                # plt.imshow(np.squeeze(repeated_slice))
                # plt.show()

    return data_out


def fix_crinfo(crinfo, to="axis"):
    """
    Function recognize order of crinfo and convert it to proper format.
    """

    crinfo = np.asarray(crinfo)
    if crinfo.shape[0] == 2:
        crinfo = crinfo.T

    return crinfo
