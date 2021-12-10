#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organ segmentation

Example:

$ pycat -f head.mat -o brain.mat
"""

from __future__ import absolute_import, division, print_function


import logging

logger = logging.getLogger(__name__)

# import unittest
# from optparse import OptionParser
import argparse
import sys

# import os.path as op


from scipy.io import loadmat
import numpy as np
import time
import scipy.stats
import copy
import pygco

# from pygco import cut_from_graph

# from . import models
from imcut.image_manipulation import (
    seed_zoom,
    zoom_to_shape,
    resize_to_shape,
    resize_to_shape_with_zoom,
    select_objects_by_seeds,
    crop,
    uncrop,
)

from imcut.models import Model, Model3D, defaultmodelparams, methods, accepted_methods
from imcut.graph import Graph
# from imcut.graph import Graph


class ImageGraphCut:
    """
    Interactive Graph Cut.

    ImageGraphCut(data, zoom, modelparams)
    scale

    Example:

    igc = ImageGraphCut(data)
    igc.interactivity()
    igc.make_gc()
    igc.show_segmentation()

    """

    def __init__(
        self,
        img,
        modelparams=None,
        segparams=None,
        voxelsize=None,
        debug_images=False,
        volume_unit="mm3",
        interactivity_loop_finish_fcn=None,
        keep_graph_properties=False,
        apriori=None
    ):
        """

        Args:
            :param img: input data
            :param modelparams: parameters of model
            :param segparams: segmentation parameters
                use_apriori_if_available - set self.apriori to ndimage with same shape as img
                apriori_gamma: influence of apriory information. 0 means no influence, 1.0 is 100% use of
                apriori information
                boundary_penalties_sigma
                pairwise_alpha_per_square_unit: the pairwise_alpha is rewriten if this parameter is used.
                    pairwise_alpha = pairwise_alpha_per_square_unit / mean(voxelsize)
                return_only_object_with_seeds: Ignore unconnected parts of segmentation, default False
            :param voxelsize: size of voxel
            :param debug_images: use to show debug images with matplotlib
            :param volume_unit: define string of volume unit. Default is "mm3"
            :param keep_graph_properties: Do not delete some usefull varibales like
                msinds, unariesalt, nlinks, temp_msgc_resized_segmentation, temp_msgc_resized_img and
                temp_msgc_resized_seeds
            :param apriori: Map with apriori information about image in range from zero to one. The weight between image
            information and apriori map is controlled by `apriori_gamma` in `segparams`.


        Returns:

        """

        logger.debug(
            "modelparams: %s, segparams: %s, voxelsize: %s, debug_images: %s",
            modelparams,
            segparams,
            voxelsize,
            debug_images,
        )
        if modelparams is None:
            modelparams = {}
        if segparams is None:
            segparams = {}
        if voxelsize is None:
            voxelsize = [1, 1, 1]

        self.voxelsize = np.asarray(voxelsize)
        if voxelsize is not None:
            self.voxel_volume = np.prod(voxelsize)
        else:
            self.voxel_volume = None

        self._update_segparams(segparams, modelparams)

        self.img = img
        self.tdata = {}
        self.segmentation = None
        # self.segparams = segparams
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)
        self.debug_images = debug_images
        self.volume_unit = volume_unit

        self.interactivity_counter = 0
        self.stats = {"tlinks shape": [], "nlinks shape": []}
        self.apriori = apriori
        self.interactivity_loop_finish_funcion = interactivity_loop_finish_fcn
        self.keep_temp_properties = keep_graph_properties

        self.msinds = None
        self.unariesalt2 = None
        self.nlinks = None

    def _update_segparams(self, segparams={}, modelparams={}):
        # default values                              use_boundary_penalties
        # self.segparams = {'pairwiseAlpha':10, 'use_boundary_penalties':False}
        self.segparams = {
            "method": "graphcut",
            "pairwise_alpha": 20,
            "use_boundary_penalties": False,
            "boundary_penalties_sigma": 200,
            "boundary_penalties_weight": 30,
            "return_only_object_with_seeds": False,
            "use_old_similarity": True,  # New similarity not implemented @TODO
            "use_extra_features_for_training": False,
            "use_apriori_if_available": True,
            "apriori_gamma": 0.1,
        }
        if "modelparams" in segparams.keys():
            modelparams = segparams["modelparams"]
        self.segparams.update(segparams)
        if "pairwise_alpha_per_square_unit" in segparams:
            self.segparams["pairwise_alpha"] = self.segparams[
                "pairwise_alpha_per_square_unit"
            ] / np.mean(self.voxel_volume)
        self.modelparams = defaultmodelparams.copy()
        self.modelparams.update(modelparams)
        self.mdl = Model(modelparams=self.modelparams)

    def load(self, filename, fv_extern=None):
        """
        Read model stored in the file.

        :param filename: Path to file with model
        :param fv_extern: external feature vector function is passed here
        :return:
        """
        self.modelparams["mdl_stored_file"] = filename
        if fv_extern is not None:
            self.modelparams["fv_extern"] = fv_extern

        # segparams['modelparams'] = {
        #     'mdl_stored_file': mdl_stored_file,
        #     # 'fv_extern': fv_function
        # }
        self.mdl = Model(modelparams=self.modelparams)

    def interactivity_loop(self, pyed):
        # @TODO stálo by za to, přehodit tlačítka na myši. Levé má teď
        # jedničku, pravé dvojku. Pravým však zpravidla označujeme pozadí a tak
        # nám vyjde popředí jako nula a pozadí jako jednička.
        # Tím také dopadne jinak interaktivní a neinteraktivní varianta.
        # import sys
        # print "logger ", logging.getLogger().getEffectiveLevel()
        # from guppy import hpy
        # h = hpy()
        # print h.heap()
        # import pdb

        # logger.debug("obj gc   " + str(sys.getsizeof(self)))

        self.set_seeds(pyed.getSeeds())
        self.run()
        pyed.setContours(1 - self.segmentation.astype(np.int8))

        # if self.interactivity_loop_finish_funcion is None:
        #     # TODO remove this statement after lisa package update (12.5.2018)
        #     from lisa import audiosupport
        #     self.interactivity_loop_finish_funcion = audiosupport.beep

        if self.interactivity_loop_finish_funcion is not None:
            self.interactivity_loop_finish_funcion()

        self.interactivity_counter += 1
        logger.debug("interactivity counter: %s", str(self.interactivity_counter))

    def __uniform_npenalty_fcn(self, orig_shape):
        return np.ones(orig_shape, dtype=np.int8)

    def __ms_npenalty_fcn(self, axis, mask, orig_shape):
        """
        :param axis: direction of edge
        :param mask: 3d ndarray with ones where is fine resolution

        Neighboorhood penalty between small pixels should be smaller then in
        bigger tiles. This is the way how to set it.

        """
        maskz = zoom_to_shape(mask, orig_shape)

        maskz_new = np.zeros(orig_shape, dtype=np.int16)
        maskz_new[maskz == 0] = self._msgc_npenalty_table[0, axis]
        maskz_new[maskz == 1] = self._msgc_npenalty_table[1, axis]
        # import sed3
        # ed = sed3.sed3(maskz_new)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        return maskz_new

    def __msgc_step0_init(self):

        # table with size 2 * self.img.ndims
        # first axis  describe whether is the edge between lowres(0) or highres(1) voxels
        # second axis describe edge direction (edge axis)
        self._msgc_npenalty_table = np.array(
            [
                [self.segparams["block_size"] * self.segparams["tile_zoom_constant"]]
                * self.img.ndim,
                [1] * self.img.ndim,
            ]
        )
        # self.__msgc_npenalty_lowres =
        # self.__msgc_npenalty_higres = 1

    def __msgc_step12_low_resolution_segmentation(self):
        """
        Get the segmentation and the
        :return:
        """
        import scipy

        start = self._start_time
        # ===== low resolution data processing
        # default parameters
        # TODO segparams_lo and segparams_hi je tam asi zbytecně
        sparams_lo = {
            "boundary_dilatation_distance": 2,
            "block_size": 6,
            "use_boundary_penalties": True,
            "boundary_penalties_weight": 1,
            "tile_zoom_constant": 1,
        }

        sparams_lo.update(self.segparams)
        sparams_hi = copy.copy(sparams_lo)
        # sparams_lo['boundary_penalties_weight'] = (
        #         sparams_lo['boundary_penalties_weight'] *
        #         sparams_lo['block_size'])
        self.segparams = sparams_lo

        self.stats["t1"] = time.time() - start
        # step 1:  low res GC
        hiseeds = self.seeds
        # ms_zoom = 4  # 0.125 #self.segparams['scale']
        # ms_zoom = self.segparams['block_size']
        # loseeds = pyed.getSeeds()
        # logger.debug("msc " + str(np.unique(hiseeds)))
        loseeds = seed_zoom(hiseeds, self.segparams["block_size"])

        hard_constraints = True

        self.seeds = loseeds

        modelparams_hi = self.modelparams.copy()
        # feature vector will be computed from selected voxels
        self.modelparams["use_extra_features_for_training"] = True

        # TODO what with voxels? It is used from here
        # hiseeds and hiimage is used to create intensity model
        self.voxels1 = self.img[hiseeds == 1].reshape(-1, 1)
        self.voxels2 = self.img[hiseeds == 2].reshape(-1, 1)
        # this is how to compute with loseeds resolution but in wrong way
        # self.voxels1 = self.img[self.seeds == 1]
        # self.voxels2 = self.img[self.seeds == 2]

        # self.voxels1 = pyed.getSeedsVal(1)
        # self.voxels2 = pyed.getSeedsVal(2)

        img_orig = self.img

        # TODO this should be done with resize_to_shape_whith_zoom
        zoom = np.asarray(loseeds.shape).astype(np.float) / img_orig.shape
        self.img = scipy.ndimage.interpolation.zoom(img_orig, zoom, order=0)
        voxelsize_orig = self.voxelsize
        logger.debug("zoom %s", zoom)
        logger.debug("vs %s", self.voxelsize)
        self.voxelsize = self.voxelsize * zoom

        # self.img = resize_to_shape_with_zoom(img_orig, loseeds.shape, 1.0 / ms_zoom, order=0)

        # this step set the self.segmentation
        self.__single_scale_gc_run()
        # logger.debug(
        #     'segmentation - max: %d min: %d' % (
        #         np.max(self.segmentation),
        #         np.min(self.segmentation)
        #     )
        # )
        logger.debug(
            "segmentation: %s", scipy.stats.describe(self.segmentation, axis=None)
        )

        self.modelparams = modelparams_hi
        self.voxelsize = voxelsize_orig
        self.img = img_orig
        self.seeds = hiseeds
        self.stats["t2"] = time.time() - start
        return hard_constraints

    def __msgc_step3_discontinuity_localization(self):
        """
        Estimate discontinuity in basis of low resolution image segmentation.
        :return: discontinuity in low resolution
        """
        import scipy

        start = self._start_time
        seg = 1 - self.segmentation.astype(np.int8)
        self.stats["low level object voxels"] = np.sum(seg)
        self.stats["low level image voxels"] = np.prod(seg.shape)
        # in seg is now stored low resolution segmentation
        # back to normal parameters
        # step 2: discontinuity localization
        # self.segparams = sparams_hi
        seg_border = scipy.ndimage.filters.laplace(seg, mode="constant")
        logger.debug("seg_border: %s", scipy.stats.describe(seg_border, axis=None))
        # logger.debug(str(np.max(seg_border)))
        # logger.debug(str(np.min(seg_border)))
        seg_border[seg_border != 0] = 1
        logger.debug("seg_border: %s", scipy.stats.describe(seg_border, axis=None))
        # scipy.ndimage.morphology.distance_transform_edt
        boundary_dilatation_distance = self.segparams["boundary_dilatation_distance"]
        seg = scipy.ndimage.morphology.binary_dilation(
            seg_border,
            # seg,
            np.ones(
                [
                    (boundary_dilatation_distance * 2) + 1,
                    (boundary_dilatation_distance * 2) + 1,
                    (boundary_dilatation_distance * 2) + 1,
                ]
            ),
        )
        if self.keep_temp_properties:
            self.temp_msgc_lowres_discontinuity = seg
        else:
            self.temp_msgc_lowres_discontinuity = None

        if self.debug_images:
            import sed3

            pd = sed3.sed3(seg_border)  # ), contour=seg)
            pd.show()
            pd = sed3.sed3(seg)  # ), contour=seg)
            pd.show()
        # segzoom = scipy.ndimage.interpolation.zoom(seg.astype('float'), zoom,
        #                                                order=0).astype('int8')
        self.stats["t3"] = time.time() - start
        return seg

    def __msgc_step45678_hi2lo_construct_graph(self, hard_constraints, seg):
        # step 4: indexes of new dual graph

        hiseeds = self.seeds
        msinds, mask_orig = self.__hi2lo_multiscale_indexes(
            seg, self.img.shape
        )  # , ms_zoom)
        logger.debug("multiscale inds " + str(msinds.shape))
        # if deb:
        #     import sed3
        #     pd = sed3.sed3(msinds, contour=seg)
        #     pd.show()

        # intensity values for indexes
        # @TODO compute average values for low resolution
        ms_img = self.img

        # @TODO __ms_create_nlinks , use __ordered_values_by_indexes
        # import pdb; pdb.set_trace() # BREAKPOINT
        # pyed.setContours(seg)

        # there is need to set correct weights between neighbooring pixels
        # this is not nice hack.
        # @TODO reorganise segparams and create_nlinks function
        # orig_shape = img_orig.shape

        self.stats["t4"] = time.time() - self._start_time

        def local_ms_npenalty(x):
            return self.__ms_npenalty_fcn(x, seg, self.img.shape)

        # here are not unique couples of nodes
        nlinks_not_unique = self.__create_nlinks(
            ms_img,
            msinds,
            # boundary_penalties_fcn=ms_npenalty_fcn
            boundary_penalties_fcn=local_ms_npenalty,
        )
        # nlinks created
        self.stats["t5"] = time.time() - self._start_time

        # get unique set
        # remove repetitive link from one pixel to another
        nlinks = ms_remove_repetitive_link(nlinks_not_unique)
        # now remove cycle link
        self.stats["t6"] = time.time() - self._start_time
        nlinks = np.array([line for line in nlinks if line[0] != line[1]])

        self.stats["t7"] = time.time() - self._start_time
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        # tlinks - indexes, data_merge
        ms_values_lin = self.__ordered_values_by_indexes(self.img, msinds)
        seeds = hiseeds
        # seeds = pyed.getSeeds()
        # if deb:
        #     import sed3
        #     se = sed3.sed3(seeds)
        #     se.show()
        ms_seeds_lin = self.__ordered_values_by_indexes(seeds, msinds)
        # logger.debug("unique seeds " + str(np.unique(seeds)))
        # logger.debug("unique seeds " + str(np.unique(ms_seeds_lin)))
        mul_mask, mul_val = self.__msgc_tlinks_area_weight_from_low_segmentation(
            mask_orig
        )
        mul_mask_lin = self.__ordered_values_by_indexes(mul_mask, msinds)
        area_weight = 1
        # TODO vyresit voxelsize
        unariesalt = self.__create_tlinks(
            ms_values_lin,
            voxelsize=self.voxelsize,
            # self.voxels1, self.voxels2,
            seeds=ms_seeds_lin,
            area_weight=area_weight,
            hard_constraints=hard_constraints,
            mul_mask=mul_mask_lin,
            mul_val=mul_val,
        )

        unariesalt2 = unariesalt.reshape(-1, 2)
        self.stats["t8"] = time.time() - self._start_time
        return nlinks, unariesalt2, msinds

    def __msgc_tlinks_area_weight_from_low_segmentation(self, loseg):
        mul_val = (self.segparams["block_size"]) ** 3
        # TODO find correct value
        # mul_val = 1
        logger.debug(
            "w: %s, loseg: %s, loseg.shape: %s",
            mul_val,
            scipy.stats.describe(loseg, axis=None),
            loseg.shape,
        )
        if loseg.shape == self.img.shape:
            loseg_resized = loseg
        else:
            # resize
            loseg_resized = zoom_to_shape(loseg, self.img.shape, dtype=np.int8)
            pass
        # area_weight = loseg_resized.astype(np.int8) * w
        mul_mask = ~loseg_resized.astype(np.bool)
        return mul_mask, mul_val

    def __msgc_step9_finish_perform_gc_and_reshape(self, nlinks, unariesalt2, msinds):
        start = self._start_time
        # create potts pairwise
        # pairwiseAlpha = -10
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.segparams["pairwise_alpha"] * pairwise).astype(np.int32)

        # print 'data shape ', img_orig.shape
        # print 'nlinks sh ', nlinks.shape
        # print 'tlinks sh ', unariesalt.shape

        # print "cut_from_graph"
        # print "unaries sh ", unariesalt.reshape(-1,2).shape
        # print "nlinks sh",  nlinks.shape
        self.stats["t9"] = time.time() - start
        self.stats["tlinks shape"].append(unariesalt2.shape)
        self.stats["nlinks shape"].append(nlinks.shape)
        start_gc = time.time()
        # Same functionality is in self.seg_data()
        result_graph = pygco.cut_from_graph(
            nlinks.astype(np.int32),
            unariesalt2.astype(np.int32),
            pairwise.astype(np.int32),
        )

        elapsed = time.time() - start_gc
        self.stats["gc time"] = elapsed
        self.stats["t10"] = time.time() - start

        # probably not necessary
        #        del nlinks
        #        del unariesalt

        # print "unaries %.3g , %.3g" % (np.max(unariesalt),np.min(unariesalt))
        # @TODO get back original data
        # result_labeling = result_graph.reshape(data.shape)
        result_labeling = result_graph[msinds]
        # import py3DSeedEditor
        # ped = py3DSeedEditor.py3DSeedEditor(result_labeling)
        # ped.show()
        result_labeling = self.__just_objects_with_seeds_if_required(result_labeling)
        self.segmentation = result_labeling
        if self.keep_temp_properties:
            self.msinds = msinds
            self.unariesalt2 = unariesalt2
            self.nlinks = nlinks
        else:
            self.msinds = None
            self.unariesalt2 = None
            self.nlinks = None

    def __multiscale_gc_lo2hi_run(self):  # , pyed):
        """
        Run Graph-Cut segmentation with refinement of low resolution multiscale graph.
        In first step is performed normal GC on low resolution data
        Second step construct finer grid on edges of segmentation from first
        step.
        There is no option for use without `use_boundary_penalties`
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        self._msgc_lo2hi_resize_init()
        self.__msgc_step0_init()

        hard_constraints = self.__msgc_step12_low_resolution_segmentation()
        # ===== high resolution data processing
        seg = self.__msgc_step3_discontinuity_localization()

        self.stats["t3.1"] = time.time() - self._start_time
        graph = Graph(
            seg,
            voxelsize=self.voxelsize,
            nsplit=self.segparams["block_size"],
            edge_weight_table=self._msgc_npenalty_table,
            compute_low_nodes_index=True,
        )

        # graph.run() = graph.generate_base_grid() + graph.split_voxels()
        # graph.run()
        graph.generate_base_grid()
        self.stats["t3.2"] = time.time() - self._start_time
        graph.split_voxels()

        self.stats["t3.3"] = time.time() - self._start_time

        self.stats.update(graph.stats)
        self.stats["t4"] = time.time() - self._start_time
        # TODO use mul_mask and mul_val
        mul_mask, mul_val = self.__msgc_tlinks_area_weight_from_low_segmentation(seg)
        area_weight = 1
        unariesalt = self.__create_tlinks(
            self.img,
            self.voxelsize,
            self.seeds,
            area_weight=area_weight,
            hard_constraints=hard_constraints,
            mul_mask=None,
            mul_val=None,
        )
        # N-links prepared
        self.stats["t5"] = time.time() - self._start_time
        un, ind = np.unique(graph.msinds, return_index=True)
        self.stats["t6"] = time.time() - self._start_time

        self.stats["t7"] = time.time() - self._start_time
        unariesalt2_lo2hi = np.hstack(
            [unariesalt[ind, 0, 0].reshape(-1, 1), unariesalt[ind, 0, 1].reshape(-1, 1)]
        )
        nlinks_lo2hi = np.hstack([graph.edges, graph.edges_weights.reshape(-1, 1)])
        if self.debug_images:
            import sed3

            ed = sed3.sed3(unariesalt[:, :, 0].reshape(self.img.shape))
            ed.show()
            import sed3

            ed = sed3.sed3(unariesalt[:, :, 1].reshape(self.img.shape))
            ed.show()
            # ed = sed3.sed3(seg)
            # ed.show()
            # import sed3
            # ed = sed3.sed3(graph.data)
            # ed.show()
            # import sed3
            # ed = sed3.sed3(graph.msinds)
            # ed.show()

        # nlinks, unariesalt2, msinds = self.__msgc_step45678_construct_graph(area_weight, hard_constraints, seg)
        # self.__msgc_step9_finish_perform_gc_and_reshape(nlinks, unariesalt2, msinds)
        self.__msgc_step9_finish_perform_gc_and_reshape(
            nlinks_lo2hi, unariesalt2_lo2hi, graph.msinds
        )
        self._msgc_lo2hi_resize_clean_finish()

    def __multiscale_gc_hi2lo_run(self):  # , pyed):
        """
        Run Graph-Cut segmentation with simplifiyng of high resolution multiscale graph.
        In first step is performed normal GC on low resolution data
        Second step construct finer grid on edges of segmentation from first
        step.
        There is no option for use without `use_boundary_penalties`
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()

        self.__msgc_step0_init()
        hard_constraints = self.__msgc_step12_low_resolution_segmentation()
        # ===== high resolution data processing
        seg = self.__msgc_step3_discontinuity_localization()
        nlinks, unariesalt2, msinds = self.__msgc_step45678_hi2lo_construct_graph(
            hard_constraints, seg
        )
        self.__msgc_step9_finish_perform_gc_and_reshape(nlinks, unariesalt2, msinds)

    def __ordered_values_by_indexes(self, data, inds):
        """
        Return values (intensities) by indexes.

        Used for multiscale graph cut.
        data = [[0 1 1],
                [0 2 2],
                [0 2 2]]

        inds = [[0 1 2],
                [3 4 4],
                [5 4 4]]

        return: [0, 1, 1, 0, 2, 0]

        If the data are not consistent, it will take the maximal value

        """
        # get unique labels and their first indexes
        # lab, linds = np.unique(inds, return_index=True)
        # compute values by indexes
        # values = data.reshape(-1)[linds]

        # alternative slow implementation
        # if there are different data on same index, it will take
        # maximal value
        # lab = np.unique(inds)
        # values = [0]*len(lab)
        # for label in lab:
        #     values[label] = np.max(data[inds == label])
        #
        # values = np.asarray(values)

        # yet another implementation
        values = [None] * (np.max(inds) + 1)

        linear_inds = inds.ravel()
        linear_data = data.ravel()
        for i in range(0, len(linear_inds)):
            # going over all data pixels

            if values[linear_inds[i]] is None:
                # this index is found for first
                values[linear_inds[i]] = linear_data[i]
            elif values[linear_inds[i]] < linear_data[i]:
                # here can be changed maximal or minimal value
                values[linear_inds[i]] = linear_data[i]

        values = np.asarray(values)

        return values

    def __hi2lo_multiscale_indexes(self, mask, orig_shape):  # , zoom):
        """
        Function computes multiscale indexes of ndarray.

        mask: Says where is original resolution (0) and where is small
        resolution (1). Mask is in small resolution.

        orig_shape: Original shape of input data.
        zoom: Usually number greater then 1

        result = [[0 1 2],
                  [3 4 4],
                  [5 4 4]]
        """

        mask_orig = zoom_to_shape(mask, orig_shape, dtype=np.int8)

        inds_small = np.arange(mask.size).reshape(mask.shape)
        inds_small_in_orig = zoom_to_shape(inds_small, orig_shape, dtype=np.int8)
        inds_orig = np.arange(np.prod(orig_shape)).reshape(orig_shape)

        # inds_orig = inds_orig * mask_orig
        inds_orig += np.max(inds_small_in_orig) + 1
        # print 'indexes'
        # import py3DSeedEditor as ped
        # import pdb; pdb.set_trace() # BREAKPOINT

        #  '==' is not the same as 'is' for numpy.array
        inds_small_in_orig[mask_orig == True] = inds_orig[mask_orig == True]  # noqa
        inds = inds_small_in_orig
        # print np.max(inds)
        # print np.min(inds)
        inds = relabel_squeeze(inds)
        logger.debug(
            "Index after relabeling: %s", scipy.stats.describe(inds, axis=None)
        )
        # logger.debug("Minimal index after relabeling: " + str(np.min(inds)))
        # inds_orig[mask_orig==True] = 0
        # inds_small_in_orig[mask_orig==False] = 0
        # inds = (inds_orig + np.max(inds_small_in_orig) + 1) + inds_small_in_orig

        return inds, mask_orig

    def interactivity(self, min_val=None, max_val=None, qt_app=None):
        """
        Interactive seed setting with 3d seed editor
        """
        from .seed_editor_qt import QTSeedEditor

        from PyQt5.QtWidgets import QApplication

        if min_val is None:
            min_val = np.min(self.img)

        if max_val is None:
            max_val = np.max(self.img)

        window_c = (max_val + min_val) / 2  # .astype(np.int16)
        window_w = max_val - min_val  # .astype(np.int16)

        if qt_app is None:
            qt_app = QApplication(sys.argv)

        pyed = QTSeedEditor(
            self.img,
            modeFun=self.interactivity_loop,
            voxelSize=self.voxelsize,
            seeds=self.seeds,
            volume_unit=self.volume_unit,
        )

        pyed.changeC(window_c)
        pyed.changeW(window_w)

        qt_app.exec_()

    def set_seeds(self, seeds):
        """
        Function for manual seed setting. Sets variable seeds and prepares
        voxels for density model.
        :param seeds: ndarray (0 - nothing, 1 - object, 2 - background,
        3 - object just hard constraints, no model training, 4 - background 
        just hard constraints, no model training)
        """
        if self.img.shape != seeds.shape:
            raise Exception("Seeds must be same size as input image")

        self.seeds = seeds.astype("int8")
        self.voxels1 = self.img[self.seeds == 1]
        self.voxels2 = self.img[self.seeds == 2]

    def run(self, run_fit_model=True):
        """
        Run the Graph Cut segmentation according to preset parameters.

        :param run_fit_model: Allow to skip model fit when the model is prepared before
        :return:
        """

        if run_fit_model:
            self.fit_model(self.img, self.voxelsize, self.seeds)

        self._start_time = time.time()
        if self.segparams["method"].lower() in ("graphcut", "gc"):
            self.__single_scale_gc_run()
        elif self.segparams["method"].lower() in (
            "multiscale_graphcut",
            "multiscale_gc",
            "msgc",
            "msgc_lo2hi",
            "lo2hi",
            "multiscale_graphcut_lo2hi",
        ):
            logger.debug("performing multiscale Graph-Cut lo2hi")
            self.__multiscale_gc_lo2hi_run()
        elif self.segparams["method"].lower() in (
            "msgc_hi2lo",
            "hi2lo",
            "multiscale_graphcut_hi2lo",
        ):
            logger.debug("performing multiscale Graph-Cut hi2lo")
            self.__multiscale_gc_hi2lo_run()
        else:
            logger.error("Unknown segmentation method: " + self.segparams["method"])

    def __single_scale_gc_run(self):
        res_segm = self._ssgc_prepare_data_and_run_computation(
            # self.img,
            #                      self
            # self.voxels1, self.voxels2,
            # seeds=self.seeds
        )
        res_segm = self.__just_objects_with_seeds_if_required(res_segm)
        self.segmentation = res_segm.astype(np.int8)

    def __just_objects_with_seeds_if_required(self, res_segm):
        # print("return_only_object_with_seeds")
        if self.segparams["return_only_object_with_seeds"]:
            logger.debug("return_only_object_with_seeds")
            try:
                # because of negative problem is as 1 segmented background and
                # as 0 is segmented foreground
                # import thresholding_functions
                # newData = thresholding_functions.getPriorityObjects(
                # newData = get_priority_objects(
                #     (1 - res_segm),
                #     nObj=-1,
                #     seeds=(self.seeds == 1).nonzero(),
                #     debug=False
                # )
                newData = select_objects_by_seeds(1 - res_segm, seeds=self.seeds)
                res_segm = 1 - newData
            except:
                import traceback

                logger.warning("Cannot import thresholding_funcions")
                traceback.print_exc()
        return res_segm

    def __set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        """
        it works with seed labels:
        0: nothing
        1: object 1 - full seeds
        2: object 2 - full seeds
        3: object 1 - not a training seeds
        4: object 2 - not a training seeds
        """
        seeds_mask = (seeds == 1) | (seeds == 3)
        tdata2[seeds_mask] = np.max(tdata2) + 1
        tdata1[seeds_mask] = 0

        seeds_mask = (seeds == 2) | (seeds == 4)
        tdata1[seeds_mask] = np.max(tdata1) + 1
        tdata2[seeds_mask] = 0

        return tdata1, tdata2

    def _boundary_penalties_array(self, axis, sigma=None):

        import scipy.ndimage.filters as scf

        # for axis in range(0,dim):
        filtered = scf.prewitt(self.img, axis=axis)
        if sigma is None:
            sigma2 = np.var(self.img)
        else:
            sigma2 = sigma ** 2

        filtered = np.exp(-np.power(filtered, 2) / (256 * sigma2))

        # srovnán hodnot tak, aby to vycházelo mezi 0 a 100
        # cc = 10
        # filtered = ((filtered - 1)*cc) + 10
        # logger.debug(
        #     'ax %.1g max %.3g min %.3g  avg %.3g' % (
        #         axis, np.max(filtered), np.min(filtered), np.mean(filtered))
        # )
        logger.debug(
            "boundary penalties, axis: %s, filtered: %s",
            axis,
            scipy.stats.describe(filtered, axis=None),
        )
        #
        # @TODO Check why forumla with exp is not stable
        # Oproti Boykov2001b tady nedělím dvojkou. Ta je tam jen proto,
        # aby to slušně vycházelo, takže jsem si jí upravil
        # Originální vzorec je
        # Bpq = exp( - (Ip - Iq)^2 / (2 * \sigma^2) ) * 1 / dist(p,q)
        #        filtered = (-np.power(filtered,2)/(16*sigma))
        # Přičítám tu 256 což je empiricky zjištěná hodnota - aby to dobře vyšlo
        # nedávám to do exponenciely, protože je to numericky nestabilní
        # filtered = filtered + 255 # - np.min(filtered2) + 1e-30
        # Ještě by tady měl a následovat exponenciela, ale s ní je to numericky
        # nestabilní. Netuším proč.
        # if dim >= 1:
        # odecitame od sebe tentyz obrazek
        # df0 = self.img[:-1,:] - self.img[]
        # diffs.insert(0,
        return filtered

    def _reshape_unariesalt_to_similarity(self, unariesalt, shape):
        # if unariesalt is None:
        #     unariesalt = self.unariesalt2
        # if shape is None:
        #     shape = self.img.shape
        #     if hasattr(self, "temp_msgc_resized_img"):
        #         shape = self.temp_msgc_resized_img.shape
        tdata1 = unariesalt[..., 0].reshape(shape)
        tdata2 = unariesalt[..., 1].reshape(shape)
        return tdata1, tdata2

    def _debug_show_unariesalt(
        self, unariesalt, suptitle=None, slice_number=None, show=True, bins=20
    ):
        shape = self.img.shape
        # print("unariesalt dtype ", unariesalt.dtype)
        tdata1, tdata2 = self._reshape_unariesalt_to_similarity(unariesalt, shape)
        self._debug_show_tdata_images(
            tdata1,
            tdata2,
            suptitle=suptitle,
            slice_number=slice_number,
            show=show,
            bins=bins,
        )

    def _debug_show_tdata_images(
        self, tdata1, tdata2, suptitle=None, slice_number=None, show=True, bins=20
    ):
        # Show model parameters
        logger.debug("tdata1 shape %s", str(tdata1.shape))
        if slice_number is None:
            slice_number = int(tdata1.shape[0] / 2)
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            if suptitle is not None:
                fig.suptitle(suptitle)
            ax = fig.add_subplot(121)
            ax.imshow(tdata1[slice_number, :, :])
            # plt.colorbar(ax=ax)

            # fig = plt.figure()
            ax = fig.add_subplot(122)
            ax.imshow(tdata2[slice_number, :, :])
            # plt.colorbar(ax=ax)

            # print('tdata1 max ', np.max(tdata1), ' min ', np.min(tdata1), " dtype ", tdata1.dtype)
            # print('tdata2 max ', np.max(tdata2), ' min ', np.min(tdata2), " dtype ", tdata2.dtype)
            # logger.debug('tdata1 max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)) + " dtype " +str( tdata1.dtype))
            logger.debug("tdata1: %s", scipy.stats.describe(tdata1, axis=None))
            logger.debug("tdata2: %s", scipy.stats.describe(tdata2, axis=None))
            # print('tdata2 max ', np.max(tdata2), ' min ', np.min(tdata2), " dtype ", tdata2.dtype)

            # # histogram
            # fig = plt.figure()
            # vx1 = data[seeds==1]
            # vx2 = data[seeds==2]
            # plt.hist([vx1, vx2], 30)

            # plt.hist(voxels2)

        except:
            import traceback

            print(traceback.format_exc())
            logger.debug("problem with showing debug images")

        try:
            fig = plt.figure()
            if suptitle is not None:
                fig.suptitle(suptitle)
            ax = fig.add_subplot(121)
            plt.hist(tdata1.flatten(), bins=bins)
            ax = fig.add_subplot(122)
            plt.hist(tdata2.flatten(), bins=bins)
        except:
            import traceback

            print(traceback.format_exc())

        if show:
            plt.show()
        return fig

    def debug_show_model(
        self, start=-1000, stop=1000, nsteps=400, suptitle=None, show=True
    ):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        if suptitle is not None:
            fig.suptitle(suptitle)
        ax = fig.add_subplot(111)
        hstx = np.linspace(start, stop, nsteps)
        ax.plot(hstx, np.exp(self.mdl.likelihood_from_image(hstx, self.voxelsize, 1)))
        ax.plot(hstx, np.exp(self.mdl.likelihood_from_image(hstx, self.voxelsize, 2)))
        if show:
            plt.show()
        return fig

    def fit_model(self, data=None, voxelsize=None, seeds=None):
        if data is None:
            data = self.img
        if voxelsize is None:
            voxelsize = self.voxelsize
        if seeds is None:
            seeds = self.seeds
        # TODO rewrite just for one class and call separatelly for obj and background.

        # TODO rename voxels1 and voxels2
        # voxe1s1 and voxels2 are used only in this function for multiscale graphcut
        # threre can be some

        # Dobře to fungovalo area_weight = 0.05 a cc = 6 a diference se
        # počítaly z :-1

        # self.mdl.trainFromSomething(data, seeds, 1, self.voxels1)
        # self.mdl.trainFromSomething(data, seeds, 2, self.voxels2)
        if self.segparams["use_extra_features_for_training"]:
            self.mdl.fit(self.voxels1, 1)
            self.mdl.fit(self.voxels2, 2)
        else:
            self.mdl.fit_from_image(data, voxelsize, seeds, [1, 2]),
        # as we convert to int, we need to multipy to get sensible values

    def __similarity_for_tlinks_obj_bgr(
        self,
        data,
        voxelsize,
        # voxels1, voxels2,
        # seeds, otherfeatures=None
    ):
        """
        Compute edge values for graph cut tlinks based on image intensity
        and texture.
        """
        # self.fit_model(data, voxelsize, seeds)
        # There is a need to have small vaues for good fit
        # R(obj) = -ln( Pr (Ip | O) )
        # R(bck) = -ln( Pr (Ip | B) )
        # Boykov2001b
        # ln is computed in likelihood
        tdata1 = (-(self.mdl.likelihood_from_image(data, voxelsize, 1))) * 10
        tdata2 = (-(self.mdl.likelihood_from_image(data, voxelsize, 2))) * 10

        # to spare some memory
        dtype = np.int16
        if np.any(tdata1 > 32760):
            dtype = np.float32
        if np.any(tdata2 > 32760):
            dtype = np.float32

        if self.segparams["use_apriori_if_available"] and self.apriori is not None:
            logger.debug("using apriori information")
            gamma = self.segparams["apriori_gamma"]
            a1 = (-np.log(self.apriori * 0.998 + 0.001)) * 10
            a2 = (-np.log(0.999 - (self.apriori * 0.998))) * 10
            # logger.debug('max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('max ' + str(np.max(a1)) + ' min ' + str(np.min(a1)))
            # logger.debug('max ' + str(np.max(a2)) + ' min ' + str(np.min(a2)))
            tdata1u = (((1 - gamma) * tdata1) + (gamma * a1)).astype(dtype)
            tdata2u = (((1 - gamma) * tdata2) + (gamma * a2)).astype(dtype)
            tdata1 = tdata1u
            tdata2 = tdata2u
            # logger.debug('   max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('   max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('gamma ' + str(gamma))

            # import sed3
            # ed = sed3.show_slices(tdata1)
            # ed = sed3.show_slices(tdata2)
            del tdata1u
            del tdata2u
            del a1
            del a2

        # if np.any(tdata1 < 0) or np.any(tdata2 <0):
        #     logger.error("Problem with tlinks. Likelihood is < 0")

        # if self.debug_images:
        #     self.__show_debug_tdata_images(tdata1, tdata2, suptitle="likelihood")
        return tdata1, tdata2

    def __limit(self, tdata1, min_limit=0, max_error=10, max_limit=20000):
        # logger.debug('before limit max ' + np.max(tdata1), 'min ' + np.min(tdata1) + " dtype " +  tdata1.dtype)
        tdata1[tdata1 > max_limit] = max_limit
        tdata1[tdata1 < min_limit] = min_limit
        # tdata1 = models.softplus(tdata1, max_error=max_error, keep_dtype=True)
        # replace inf with large finite number
        # tdata1 = np.nan_to_num(tdata1)
        return tdata1

    def __limit_tlinks(self, tdata1, tdata2):
        tdata1 = self.__limit(tdata1)
        tdata2 = self.__limit(tdata2)

        return tdata1, tdata2

    def __create_tlinks(
        self,
        data,
        voxelsize,
        # voxels1, voxels2,
        seeds,
        area_weight,
        hard_constraints,
        mul_mask=None,
        mul_val=None,
    ):
        tdata1, tdata2 = self.__similarity_for_tlinks_obj_bgr(
            data,
            voxelsize,
            # voxels1, voxels2,
            # seeds
        )

        # logger.debug('tdata1 min %f , max %f' % (tdata1.min(), tdata1.max()))
        # logger.debug('tdata2 min %f , max %f' % (tdata2.min(), tdata2.max()))
        if hard_constraints:
            if type(seeds) == "bool":
                raise Exception(
                    "Seeds variable  not set",
                    "There is need set seed if you want use hard constraints",
                )
            tdata1, tdata2 = self.__set_hard_hard_constraints(tdata1, tdata2, seeds)

        limit = 20000
        # carefull multiplication with limit to
        if mul_mask is not None:
            divided_limit = limit / mul_val
            mm = tdata1 > divided_limit
            tdata1[mul_mask & mm] = limit
            tdata1[mul_mask & ~mm] *= mul_val
            mm = tdata2 > divided_limit
            tdata2[mul_mask & mm] = limit
            tdata2[mul_mask & ~mm] *= mul_val
        # if not np.isscalar(area_weight):
        #     area_weight = area_weight.reshape(tdata1.shape)
        tdata1 = self.__limit(tdata1)
        tdata2 = self.__limit(tdata2)
        unariesalt = (
            0
            + (
                np.dstack(
                    area_weight * [tdata1.reshape(-1, 1), tdata2.reshape(-1, 1)]
                ).copy("C")
            )
        ).astype(np.int32)
        unariesalt = self.__limit(unariesalt)
        # if self.debug_images:
        #     self.__show_debug_(unariesalt, suptitle="after weighing and limitation")
        return unariesalt

    def __create_nlinks(self, data, inds=None, boundary_penalties_fcn=None):
        """
        Compute nlinks grid from data shape information. For boundary penalties
        are data (intensities) values are used.

        ins: Default is None. Used for multiscale GC. This are indexes of
        multiscale pixels. Next example shows one superpixel witn index 2.
        inds = [
            [1 2 2],
            [3 2 2],
            [4 5 6]]

        boundary_penalties_fcn: is function with one argument - axis. It can
            it can be used for setting penalty weights between neighbooring
            pixels.

        """
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        start = time.time()
        if inds is None:
            inds = np.arange(data.size).reshape(data.shape)
        # if not self.segparams['use_boundary_penalties'] and \
        #         boundary_penalties_fcn is None :
        if boundary_penalties_fcn is None:
            edgs_arr = self._prepare_edgs_arr_with_no_fcn(inds)
        else:
            logger.info("use_boundary_penalties")
            edgs_arr = self._prepare_edgs_arr_from_boundary_fcn(inds, boundary_penalties_fcn)

        # import pdb; pdb.set_trace()
        edges = np.vstack(edgs_arr).astype(np.int32)
        # edges - seznam indexu hran, kteres spolu sousedi\
        elapsed = time.time() - start
        self.stats["_create_nlinks time"] = elapsed
        logger.info("__create nlinks time " + str(elapsed))
        return edges

    def _prepare_edgs_arr_with_no_fcn(self, inds):
        if self.img.ndim == 3:
            # This is faster for some specific format
            edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            edgs_arr = [edgx, edgy, edgz]
        elif self.img.ndim == 2:
            edgx = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            edgy = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            edgs_arr = [edgx, edgy]
        else:
            logger.error(f"Input data dimension {self.img.ndim} is no supported")
            edgs_arr = None
        return edgs_arr

    def _prepare_edgs_arr_from_boundary_fcn(self, inds, boundary_penalties_fcn):
        """
        prepare list of edgs for each axis
        :param boundary_penalties_fcn: function working with intensities
        :param inds:
        :return:
        """
        bpw = self.segparams["boundary_penalties_weight"]
        if self.img.ndim == 3:

            bpa = boundary_penalties_fcn(2)
            # id1=inds[:, :, :-1].ravel()
            edgx = np.c_[
                inds[:, :, :-1].ravel(),
                inds[:, :, 1:].ravel(),
                # cc * np.ones(id1.shape)
                bpw * bpa[:, :, 1:].ravel(),
            ]

            bpa = boundary_penalties_fcn(1)
            # id1 =inds[:, 1:, :].ravel()
            edgy = np.c_[
                inds[:, :-1, :].ravel(),
                inds[:, 1:, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[:, 1:, :].ravel(),
            ]

            bpa = boundary_penalties_fcn(0)
            # id1 = inds[1:, :, :].ravel()
            edgz = np.c_[
                inds[:-1, :, :].ravel(),
                inds[1:, :, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[1:, :, :].ravel(),
            ]
            edgs_arr = [edgx, edgy, edgz]
        elif self.img.ndim == 2:
            bpa = boundary_penalties_fcn(1)
            # id1=inds[:, :, :-1].ravel()
            edgx = np.c_[
                inds[:, :-1].ravel(),
                inds[:, 1:].ravel(),
                # cc * np.ones(id1.shape)
                bpw * bpa[:, 1:].ravel(),
            ]

            bpa = boundary_penalties_fcn(0)
            # id1 =inds[:, 1:, :].ravel()
            edgy = np.c_[
                inds[:-1, :].ravel(),
                inds[1:, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[1:, :].ravel(),
            ]

            edgs_arr = [edgx, edgy]
        else:
            logger.error(f"Input data dimension {self.img.ndim} is no supported")
            edgs_arr = None
        return edgs_arr

    def debug_get_reconstructed_similarity(
            self,
            data3d=None,
            voxelsize=None,
            seeds=None,
            area_weight=1,
            hard_constraints=True,
            return_unariesalt=False,
    ):
        """
        Use actual model to calculate similarity. If no input is given the last image is used.
        :param data3d:
        :param voxelsize:
        :param seeds:
        :param area_weight:
        :param hard_constraints:
        :param return_unariesalt:
        :return:
        """
        if data3d is None:
            data3d = self.img
        if voxelsize is None:
            voxelsize = self.voxelsize
        if seeds is None:
            seeds = self.seeds

        unariesalt = self.__create_tlinks(
            data3d,
            voxelsize,
            # voxels1, voxels2,
            seeds,
            area_weight,
            hard_constraints,
        )
        if return_unariesalt:
            return unariesalt
        else:
            return self._reshape_unariesalt_to_similarity(unariesalt, data3d.shape)

    def debug_show_reconstructed_similarity(
        self,
        data3d=None,
        voxelsize=None,
        seeds=None,
        area_weight=1,
        hard_constraints=True,
        show=True,
        bins=20,
        slice_number=None,
    ):
        """
        Show tlinks.
        :param data3d: ndarray with input data
        :param voxelsize:
        :param seeds:
        :param area_weight:
        :param hard_constraints:
        :param show:
        :param bins: histogram bins number
        :param slice_number:
        :return:
        """

        unariesalt = self.debug_get_reconstructed_similarity(
            data3d,
            voxelsize=voxelsize,
            seeds=seeds,
            area_weight=area_weight,
            hard_constraints=hard_constraints,
            return_unariesalt=True,
        )

        self._debug_show_unariesalt(
            unariesalt, show=show, bins=bins, slice_number=slice_number
        )

    def debug_inspect_node(self, node_msindex):
        """
        Get info about the node. See pycut.inspect_node() for details.
        Processing is done in temporary shape.

        :param node_seed:
        :return: node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds
        """
        return inspect_node(self.nlinks, self.unariesalt2, self.msinds, node_msindex)

    def debug_get_node_msindex(self, node_seeds):
        """
        Get multiscale index of voxel selected by seeds.
        :param node_seeds: ndarray
        :return int with multiscale index

        """
        node_msindex = get_node_msindex(self.msinds, node_seeds)
        return node_msindex

    def debug_interactive_inspect_node(self):
        """
        Call after segmentation to see selected node neighborhood.
        User have to select one node by click.
        :return:
        """
        if (
            np.sum(
                np.abs(
                    np.asarray(self.msinds.shape) - np.asarray(self.segmentation.shape)
                )
            )
            == 0
        ):
            segmentation = self.segmentation
        else:
            segmentation = self.temp_msgc_resized_segmentation

        logger.info("Click to select one voxel of interest")
        import sed3

        ed = sed3.sed3(self.msinds, contour=segmentation == 0)
        ed.show()
        edseeds = ed.seeds
        node_msindex = get_node_msindex(self.msinds, edseeds)

        node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = self.debug_inspect_node(
            node_msindex
        )
        import sed3

        ed = sed3.sed3(
            self.msinds, contour=segmentation == 0, seeds=node_neighboor_seeds
        )
        ed.show()

        return (
            node_unariesalt,
            node_neighboor_edges_and_weights,
            node_neighboor_seeds,
            node_msindex,
        )

    def debug_reconstruct_nlinks_max(self):
        """
        It will shoe the maximum weight between neighboor pixels.
        It should be different in hires area and lowres area.
        In lowres area it whould be k* higher than highres where k is tile size.
        :return:
        """
        return reconstruct_nlinks_max(self.nlinks, self.msinds)

    def _ssgc_prepare_data_and_run_computation(
        self,
        # voxels1, voxels2,
        hard_constraints=True,
        area_weight=1,
    ):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace() # BREAKPOINT

        unariesalt = self.__create_tlinks(
            self.img,
            self.voxelsize,
            # voxels1, voxels2,
            self.seeds,
            area_weight,
            hard_constraints,
        )
        #  některém testu  organ semgmentation dosahují unaries -15. což je podiné
        # stačí vyhodit print před if a je to vidět
        logger.debug("unaries %.3g , %.3g" % (np.max(unariesalt), np.min(unariesalt)))
        # create potts pairwise
        # pairwiseAlpha = -10
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.segparams["pairwise_alpha"] * pairwise).astype(np.int32)
        # pairwise = np.array([[0,30],[30,0]]).astype(np.int32)
        # print pairwise

        self.iparams = {}

        if self.segparams["use_boundary_penalties"]:
            sigma = self.segparams["boundary_penalties_sigma"]
            # set boundary penalties function
            # Default are penalties based on intensity differences
            boundary_penalties_fcn = lambda ax: self._boundary_penalties_array(
                axis=ax, sigma=sigma
            )
        else:
            boundary_penalties_fcn = None
        nlinks = self.__create_nlinks(
            self.img, boundary_penalties_fcn=boundary_penalties_fcn
        )

        self.stats["tlinks shape"].append(unariesalt.reshape(-1, 2).shape)
        self.stats["nlinks shape"].append(nlinks.shape)
        # we flatten the unaries
        # result_graph = cut_from_graph(nlinks, unaries.reshape(-1, 2),
        # pairwise)
        start = time.time()
        if self.debug_images:
            self._debug_show_unariesalt(unariesalt)
        result_graph = pygco.cut_from_graph(nlinks, unariesalt.reshape(-1, 2), pairwise)
        elapsed = time.time() - start
        self.stats["gc time"] = elapsed
        result_labeling = result_graph.reshape(self.img.shape)

        return result_labeling

    def _msgc_lo2hi_resize_init(self):
        self._lo2hi_resize_original_shape = self.img.shape
        new_shape = (
            np.ceil(np.asarray(self.img.shape) / float(self.segparams["block_size"]))
            * self.segparams["block_size"]
        ).astype(np.int)
        crinfo = list(zip([0] * self.img.ndim, self.img.shape))
        self.img = uncrop(self.img, crinfo, new_shape, outside_mode="nearest")
        self.seeds = uncrop(self.seeds, crinfo, new_shape)

    def _msgc_lo2hi_resize_clean_finish(self):
        orig_shape = self._lo2hi_resize_original_shape
        self.temp_msgc_resized_img = self.img
        self.temp_msgc_resized_segmentation = self.segmentation
        self.temp_msgc_resized_seeds = self.seeds
        self.img = self.temp_msgc_resized_img[
            : orig_shape[0], : orig_shape[1], : orig_shape[2]
        ]
        self.segmentation = self.temp_msgc_resized_segmentation[
            : orig_shape[0], : orig_shape[1], : orig_shape[2]
        ]
        self.seeds = self.temp_msgc_resized_seeds[
            : orig_shape[0], : orig_shape[1], : orig_shape[2]
        ]
        if not self.keep_temp_properties:
            del self.temp_msgc_resized_segmentation
            del self.temp_msgc_resized_img
            del self.temp_msgc_resized_seeds
            del self._lo2hi_resize_original_shape
        # self.stats["t11"] = time.time() - self._start_time

    def save(self, filename):
        """
        Save model to file
        :param filename: Path to file
        :return:
        """
        self.mdl.save(filename)


def ms_remove_repetitive_link(nlinks_not_unique):
    # nlinks = np.array(
    #     [list(x) for x in set(tuple(x) for x in nlinks_not_unique)]
    # )
    a = nlinks_not_unique
    nlinks = (
        np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))
        .view(a.dtype)
        .reshape(-1, a.shape[1])
    )

    return nlinks


def get_neighborhood_edes(nlinks, node_msindex):
    node_neighbor_edges = np.vstack(
        [
            nlinks[np.where(nlinks[:, 0] == node_msindex)],
            nlinks[np.where(nlinks[:, 1] == node_msindex)],
        ]
    )
    return node_neighbor_edges


def inspect_node_neighborhood(nlinks, msinds, node_msindex):
    """
    Get information about one node in graph

    :param nlinks: neighboorhood edges
    :param msinds: indexes in 3d image
    :param node_msindex: int, multiscale index of selected voxel
    :return: node_neighboor_edges_and_weights, node_neighboor_seeds
    """
    # seed_indexes = np.nonzero(node_seed)
    # selected_inds = msinds[seed_indexes]
    # node_msindex = selected_inds[0]

    node_neighbor_edges = get_neighborhood_edes(nlinks, node_msindex)

    node_neighbor_seeds = np.zeros_like(msinds, dtype=np.int8)
    for neighboor_ind in np.unique(node_neighbor_edges[:, :2].ravel()):
        node_neighbor_ind = np.where(msinds == neighboor_ind)
        node_neighbor_seeds[node_neighbor_ind] = 2

    node_neighbor_seeds[np.where(msinds == node_msindex)] = 1

    # node_coordinates = np.unravel_index(selected_voxel_ind, msinds.shape)
    # node_neighbor_coordinates = np.unravel_index(np.unique(node_neighbor_edges[:, :2].ravel()), msinds.shape)
    return node_neighbor_edges, node_neighbor_seeds


def inspect_node(nlinks, unariesalt, msinds, node_msindex):
    """
    Get information about one node in graph

    :param nlinks: neighboorhood edges
    :param unariesalt: weights
    :param msinds: indexes in 3d image
    :param node_msindex: msindex of selected node. See get_node_msindex()
    :return: node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds
    """

    node_unariesalt = unariesalt[node_msindex]
    neigh_edges, neigh_seeds = inspect_node_neighborhood(nlinks, msinds, node_msindex)
    return node_unariesalt, neigh_edges, neigh_seeds


def get_node_msindex(msinds, node_seed):
    """
    Convert seeds-like selection of voxel to multiscale index.
    :param msinds: ndarray with indexes
    :param node_seed: ndarray with 1 where selected pixel is, or list of indexes in this array
    :return: multiscale index of first found seed
    """
    if type(node_seed) == np.ndarray:
        seed_indexes = np.nonzero(node_seed)
    elif type(node_seed) == list:
        seed_indexes = node_seed
    else:
        seed_indexes = [node_seed]

    selected_nodes_msinds = msinds[seed_indexes]
    node_msindex = selected_nodes_msinds[0]
    return node_msindex


def reconstruct_nlinks_max(nlinks, msinds):
    nlinks_max = np.zeros_like(msinds, dtype=nlinks.dtype)
    for node_msindex in np.unique(msinds):
        edges = get_neighborhood_edes(nlinks, node_msindex)
        nlinks_max[msinds == node_msindex] = np.max(edges[:, 2])
    return nlinks_max


# class Tests(unittest.TestCase):
#     def setUp(self):
#         pass

#     def test_segmentation(self):
#         data_shp = [16,16,16]
#         data = generate_data(data_shp)
#         seeds = np.zeros(data_shp)
# setting background seeds
#         seeds[:,0,0] = 1
#         seeds[6,8:-5,2] = 2
# x[4:-4, 6:-2, 1:-6] = -1

#         igc = ImageGraphCut(data)
# igc.interactivity()
# instead of interacitivity just set seeeds
#         igc.noninteractivity(seeds)

# instead of showing just test results
# igc.show_segmentation()
#         segmentation = igc.segmentation
# Testin some pixels for result
#         self.assertTrue(segmentation[0, 0, -1] == 0)
#         self.assertTrue(segmentation[7, 9, 3] == 1)
#         self.assertTrue(np.sum(segmentation) > 10)
# pdb.set_trace()
# self.assertTrue(True)


# logger.debug(igc.segmentation.shape)

usage = "%prog [options]\n" + __doc__.rstrip()
help = {
    "in_file": 'input *.mat file with "data" field',
    "out_file": "store the output matrix to the file",
    "debug": "debug mode",
    "debug_interactivity": "turn on interactive debug mode",
    "test": "run unit test",
}


def relabel_squeeze(data):
    """  Makes relabeling of data if there are unused values.  """
    palette, index = np.unique(data, return_inverse=True)
    data = index.reshape(data.shape)
    # realy slow solution
    #        unq = np.unique(data)
    #        actual_label = 0
    #        for lab in unq:
    #            data[data == lab] = actual_label
    #            actual_label += 1

    # one another solution probably slower
    # arr = data
    # data = (np.digitize(arr.reshape(-1,),np.unique(arr))-1).reshape(arr.shape)

    return data


# @profile


def main():
    # logger = logging.getLogger(__file__)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format="%(message)s")
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # parser = OptionParser(description='Organ segmentation')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--debug", action="store_true", help=help["debug"])
    parser.add_argument(
        "-di",
        "--debug-interactivity",
        action="store_true",
        help=help["debug_interactivity"],
    )
    parser.add_argument(
        "-i", "--input-file", action="store", default=None, help=help["in_file"]
    )
    parser.add_argument("-t", "--tests", action="store_true", help=help["test"])
    parser.add_argument(
        "-o",
        "--outputfile",
        action="store",
        dest="out_filename",
        default="output.mat",
        help=help["out_file"],
    )
    # (options, args) = parser.parse_args()
    options = parser.parse_args()

    debug_images = False

    if options.debug:
        logger.setLevel(logging.DEBUG)
        # print DEBUG
        # DEBUG = True

    if options.debug_interactivity:
        debug_images = True

    # if options.tests:
    #     sys.argv[1:]=[]
    #     unittest.main()

    if options.input_file is None:
        raise IOError("No input data!")

    else:
        dataraw = loadmat(options.input_file, variable_names=["data", "voxelsize_mm"])
    # import pdb; pdb.set_trace() # BREAKPOINT

    logger.debug("\nvoxelsize_mm " + dataraw["voxelsize_mm"].__str__())

    if sys.platform == "win32":
        # hack, on windows is voxelsize read as 2D array like [[1, 0.5, 0.5]]
        dataraw["voxelsize_mm"] = dataraw["voxelsize_mm"][0]

    igc = ImageGraphCut(
        dataraw["data"],
        voxelsize=dataraw["voxelsize_mm"],
        debug_images=debug_images  # noqa
        # , modelparams={'type': 'gaussian_kde', 'params': []}
        # , modelparams={'type':'kernel', 'params':[]}  #noqa not in  old scipy
        # , modelparams={'type':'gmmsame', 'params':{'cvtype':'full', 'n_components':3}} # noqa 3 components
        # , segparams={'type': 'multiscale_gc'}  # multisc gc
        ,
        segparams={"method": "multiscale_graphcut"}  # multisc gc
        # , modelparams={'fv_type': 'fv001'}
        # , modelparams={'type': 'dpgmm', 'params': {'cvtype': 'full', 'n_components': 5, 'alpha': 10}}  # noqa 3 components
    )
    igc.interactivity()

    logger.debug("igc interactivity countr: %s", igc.interactivity_counter)

    logger.debug(igc.segmentation.shape)


if __name__ == "__main__":
    main()
