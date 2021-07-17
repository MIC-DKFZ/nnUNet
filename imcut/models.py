#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

import numpy as np

import os.path as op
import sklearn
import sklearn.mixture

# version comparison
from pkg_resources import parse_version
import scipy.ndimage
import scipy.stats
from . import features

if parse_version(sklearn.__version__) > parse_version("0.10"):
    # new versions
    gmm__cvtype = "covariance_type"
    gmm__cvtype_bad = "cvtype"
    defaultmodelparams = {
        "type": "gmmsame",
        "params": {"covariance_type": "full"},
        "fv_type": "intensity",
    }
else:
    gmm__cvtype = "cvtype"
    gmm__cvtype_bad = "covariance_type"
    defaultmodelparams = {
        "type": "gmmsame",
        "params": {"cvtype": "full"},
        "fv_type": "intensity",
    }
methods = ["graphcut", "multiscale_graphcut_lo2hi", "multiscale_graphcut_hi2lo"]
accepted_methods = [
    "graphcut",
    "gc",
    "multiscale_graphcut",
    "multiscale_gc",
    "msgc",
    "msgc_lo2hi",
    "lo2hi",
    "multiscale_graphcut_lo2hi",
    "msgc_hi2lo",
    "hi2lo",
    "multiscale_graphcut_hi2lo",
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softplus(x, max_error=1, keep_dtype=True):
    x = np.asarray(x)
    dtype = x.dtype
    result = max_error * np.log(1 + np.exp(x / max_error))

    if keep_dtype:
        result = result.astype(dtype)
    return result


class Model3D(object):

    """ Model for image intensity. Last dimension represent feature vector.
    m = Model()
    m.train(cla, clb)
    X = numpy.random.random([2,3,4])
    # we have data 2x3 with fature vector with 4 fatures
    m.likelihood(X,0)

    modelparams['type']: type of model estimation. Gaussian mixture from EM
    algorithm is implemented as 'gmmsame'. Gaussian kernel density estimation
    is implemented as 'gaussian_kde'. General kernel estimation ('kernel')
    is from scipy version 0.14 and it is not tested.

    fv_type: feature vector type is defined with one of fallowing string
        intensity - based on seeds and data the intensity as feature vector is used
        voxel - information in voxel1 and voxel2 is used
        fv_extern - external feature vector function specified in fv_extern label
        fv001 - pixel and gaussian blur

    fv_extern:
        function `fv_function(data, voxelsize, seeds, unique_cls)`. It is used only
        if fv_type is set to "fv_extern"

    mdl_stored_file:
        string or False. Default is false. The string is path to file with stored model.
        This model is loaded and

    adaptation:
        - retrain: no adaptatin
        - original_data: train every class only once


    """

    def __init__(self, modelparams):
        # modelparams = {}
        # modelparams.update(parameters['modelparams'])
        if "params" in modelparams.keys() and gmm__cvtype_bad in modelparams["params"]:
            value = modelparams["params"].pop(gmm__cvtype_bad)
            modelparams["params"][gmm__cvtype] = value

        self.mdl = {}
        self.modelparams = defaultmodelparams.copy()
        self.modelparams.update({"adaptation": "retrain"})
        # if modelparams are updated after load, there are problems with some setting comming from outside and rewriting
        # for example "fv_type" into "intensity"
        self.modelparams.update(modelparams)
        if "mdl_stored_file" in modelparams.keys() and modelparams["mdl_stored_file"]:
            mdl_file = modelparams["mdl_stored_file"]
            self.load(mdl_file)

    def fit_from_image(self, data, voxelsize, seeds, unique_cls):
        """
        This Method allows computes feature vector and train model.

        :cls: list of index number of requested classes in seeds
        """
        fvs, clsselected = self.features_from_image(data, voxelsize, seeds, unique_cls)
        self.fit(fvs, clsselected)
        # import pdb
        # pdb.set_trace()
        # for fv, cl in zip(fvs, cls):
        #     fvs, clsselected = self.features_from_image(data, voxelsize, seeds, cl)
        #     logger.debug('cl: ' + str(cl))
        #     self.train(fv, cl)

    def save(self, filename):
        """
        Save model to pickle file. External feature function is not stored
        """
        import dill

        tmpmodelparams = self.modelparams.copy()
        # fv_extern_src = None
        fv_extern_name = None
        # try:
        #     fv_extern_src = dill.source.getsource(tmpmodelparams['fv_extern'])
        #     tmpmodelparams.pop('fv_extern')
        # except:
        #     pass

        # fv_extern_name = dill.source.getname(tmpmodelparams['fv_extern'])
        if "fv_extern" in tmpmodelparams:
            tmpmodelparams.pop("fv_extern")

        sv = {
            "modelparams": tmpmodelparams,
            "mdl": self.mdl,
            # 'fv_extern_src': fv_extern_src,
            # 'fv_extern_src_name': fv_extern_src_name,
            # 'fv_extern_name': fv_extern_src_name,
            #
        }
        sss = dill.dumps(self.modelparams)
        logger.debug("pickled " + str(sss))

        dill.dump(sv, open(filename, "wb"))

    def load(self, mdl_file):
        """
        load model from file. fv_type is not set with this function. It is expected to set it before.
        """
        import dill as pickle

        mdl_file_e = op.expanduser(mdl_file)

        sv = pickle.load(open(mdl_file_e, "rb"))
        self.mdl = sv["mdl"]
        # self.mdl[2] = self.mdl[0]
        # try:
        #     eval(sv['fv_extern_src'])
        #     eval("fv_extern_temp_name  = " + sv['fv_extern_src_name'])
        #     sv['fv_extern'] = fv_extern_temp_name
        # except:
        #     print "pomoc,necoje blbe"
        #     pass

        self.modelparams.update(sv["modelparams"])
        logger.debug("loaded model from path: " + mdl_file_e)
        # from PyQt4 import QtCore; QtCore.pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()

    def likelihood_from_image(self, data, voxelsize, cl):
        sha = data.shape

        likel = self.likelihood(self.features_from_image(data, voxelsize), cl)
        return likel.reshape(sha)


class Model(Model3D):
    # def __init__(self, nObjects=2, modelparams={}):
    # super(Model3D, self).__init__()

    # fix change of cvtype and covariancetype
    # print modelparams

    def features_from_image(
        self, data, voxelsize, seeds=None, unique_cls=None
    ):  # , voxels=None):
        """
        Input data is 3d image

        :param data: is 3d image
        :param seeds: ndimage with same shape as data, nonzero values means seeds.
        :param unique_cls: can select only fv for seeds from specific class.
        f.e. unique_cls = [1, 2] ignores label 0

        funcion is called twice in graph cut
        first call is with all params, second is only with data.

        based on self.modelparams['fv_type'] the feature vector is computed
        keywords "intensity", "voxels", "fv001", "fv_extern"  can be used.
        modelparams['fv_type'] = 'fv_extern' allows to use external fv function

        Example of exter feature function. For easier implementation of return values use function return_fv_by_seeds().

        def fv_function(data, voxelsize, seeds=None, cl=None):
            data2 = scipy.ndimage.filters.gaussian_filter(data, sigma=5)
            arrs = [data.reshape(-1, 1), data2.reshape(-1, 1)]
            fv = np.concatenate(arrs, axis=1)
            return imcut.features.return_fv_by_seeds(fv, seeds, unique_cls)

        modelparams['fv_extern'] = fv_function
        """

        fv_type = self.modelparams["fv_type"]
        logger.debug("fv_type " + fv_type)
        fv = []
        if fv_type == "intensity":
            fv = data.reshape(-1, 1)

            if seeds is not None:
                logger.debug("seeds: %s", scipy.stats.describe(seeds, axis=None))
                sd = seeds.reshape(-1, 1)
                selection = np.in1d(sd, unique_cls)
                fv = fv[selection]
                sd = sd[selection]
                # sd = sd[]
                return fv, sd
            return fv

        # elif fv_type in ("voxels"):
        #     if seeds is not None:
        #         fv = np.asarray(voxels).reshape(-1, 1)
        #     else:
        #         fv = data
        #         fv = fv.reshape(-1, 1)
        elif fv_type in ("fv001", "FV001", "intensity_and_blur"):

            # intensity in pixel, gaussian blur intensity
            return features.fv_function_intensity_and_smoothing(
                data, voxelsize, seeds, unique_cls
            )

            # from PyQt4.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()

            # print fv1.shape
            # print fv2.shape
            # print fv.shape
        elif fv_type == "fv_extern":
            fv_function = self.modelparams["fv_extern"]
            return fv_function(data, voxelsize, seeds, unique_cls)

        else:
            logger.error("Unknown feature vector type: " + self.modelparams["fv_type"])
        return fv

    # def trainFromSomething(self, data, seeds, cls, voxels):
    #     """
    #     This Method allows computes feature vector and train model.
    #
    #     :cl: scalar index number of class
    #     """
    #     for cl, voxels_i in zip(cls, voxels):
    #         logger.debug('cl: ' + str(cl))
    #         fv = self.createFV(data, seeds, cl, voxels_i)
    #         self.train(fv, cl)

    def fit(self, clx, cla):
        """

        Args:
            clx: feature vector
            cl: class, scalar or array

        Returns:

        """
        # TODO for now only sclar is used. Do not use scalar cl if future.
        # Model is not trained from other class konwledge
        # use model trained by all classes number.
        if np.isscalar(cla):
            self._fit_one_class(clx, cla)
        else:
            cla = np.asarray(cla)
            clx = np.asarray(clx)
            # import pdb
            # pdb.set_trace()
            for cli in np.unique(cla):
                selection = cla == cli
                clxsel = clx[np.nonzero(selection)[0]]
                self._fit_one_class(clxsel, cli)

    def _fit_one_class(self, clx, cl):
        """ Train clas number cl with data clx.

        Use trainFromImageAndSeeds() function if you want to use 3D image data
        as an input.

        clx: data, 2d matrix
        cl: label, integer

        label: gmmsame, gaussian_kde, dpgmm, stored
        """

        logger.debug("clx " + str(clx[:10, :]))
        logger.debug("clx type" + str(clx.dtype))
        # name = 'clx' + str(cl) + '.npy'
        # print name
        # np.save(name, clx)
        logger.debug("_fit()")
        if self.modelparams["adaptation"] == "original_data":
            if cl in self.mdl.keys():
                return
        # if True:
        #     return

        logger.debug("training continues")

        if self.modelparams["type"] == "gmmsame":
            if len(clx.shape) == 1:
                logger.warning(
                    "reshaping in train will be removed. Use \
                                \ntrainFromImageAndSeeds() function"
                )

                print("Warning deprecated feature in train() function")
                #  je to jen jednorozměrný vektor, tak je potřeba to
                # převést na 2d matici
                clx = clx.reshape(-1, 1)
            gmmparams = self.modelparams["params"]
            self.mdl[cl] = sklearn.mixture.GaussianMixture(**gmmparams)
            self.mdl[cl].fit(clx)

        elif self.modelparams["type"] == "kernel":
            # Not working (probably) in old versions of scikits
            # from sklearn.neighbors.kde import KernelDensity
            from sklearn.neighbors import KernelDensity

            # kernelmodelparams = {'kernel': 'gaussian', 'bandwidth': 0.2}
            kernelmodelparams = self.modelparams["params"]
            self.mdl[cl] = KernelDensity(**kernelmodelparams).fit(clx)
        elif self.modelparams["type"] == "gaussian_kde":
            # print clx
            import scipy.stats

            # from PyQt4.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()

            # gaussian_kde works only with floating point types
            self.mdl[cl] = scipy.stats.gaussian_kde(clx.astype(np.float))
        elif self.modelparams["type"] == "dpgmm":
            # print 'clx.shape ', clx.shape
            # print 'cl ', cl
            gmmparams = self.modelparams["params"]
            self.mdl[cl] = sklearn.mixture.DPGMM(**gmmparams)
            # todo here is a hack
            # dpgmm z nějakého důvodu nefunguje pro naše data
            # vždy natrénuje jednu složku v blízkosti nuly
            # patrně to bude mít něco společného s parametrem alpha
            # přenásobí-li se to malým číslem, zázračně to chodí
            self.mdl[cl].fit(clx * 0.001)
        elif self.modelparams["type"] == "stored":
            # Classifer is trained before segmentation and stored to pickle
            import pickle

            print("stored")
            logger.warning("deprecated use of stored parameters")

            mdl_file = self.modelparams["params"]["mdl_file"]
            self.mdl = pickle.load(open(mdl_file, "rb"))

        elif type(self.modelparams['type'] == 'custom'):
            self.mdl[cl].fit(clx)
        else:
            raise NameError("Unknown model type")

            # pdb.set_trace();
            # TODO remove saving
            #         self.save('classif.p')

    def likelihood(self, x, cl):
        """
        X = numpy.random.random([2,3,4])
        # we have data 2x3 with fature vector with 4 fatures

        Use likelihoodFromImage() function for 3d image input
        m.likelihood(X,0)
        """

        # sha = x.shape
        # xr = x.reshape(-1, sha[-1])
        # outsha = sha[:-1]
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        logger.debug("likel " + str(x.shape))
        if self.modelparams["type"] == "gmmsame":

            px = self.mdl[cl].score_samples(x)

        # todo ošetřit více dimenzionální fv
        # px = px.reshape(outsha)
        elif self.modelparams["type"] == "kernel":
            px = self.mdl[cl].score_samples(x)
        elif self.modelparams["type"] == "gaussian_kde":
            # print x
            # np.log because it is likelihood
            # @TODO Zde je patrně problém s reshape
            # old
            # px = np.log(self.mdl[cl](x.reshape(-1)))
            # new
            px = np.log(self.mdl[cl](x))
            # px = px.reshape(outsha)
            # from PyQt4.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()
        elif self.modelparams["type"] == "dpgmm":
            # todo here is a hack
            # dpgmm z nějakého důvodu nefunguje pro naše data
            # vždy natrénuje jednu složku v blízkosti nuly
            # patrně to bude mít něco společného s parametrem alpha
            # přenásobí-li se to malým číslem, zázračně to chodí
            logger.warning(".score() replaced with .score_samples() . Check it.")
            # px = self.mdl[cl].score(x * 0.01)
            px = self.mdl[cl].score_samples(x * 0.01)
        elif self.modelparams["type"] == "stored":
            px = self.mdl[cl].score(x)
        elif self.modelparams["type"] == "custom":
            px = self.mdl[cl].score_samples(x)
        else:
            logger.error(f"Unknown type {self.modelparams['type']}")
        return px
