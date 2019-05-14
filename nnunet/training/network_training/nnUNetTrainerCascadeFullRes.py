from multiprocessing.pool import Pool
import matplotlib
from nnunet.training.data_augmentation.default_data_augmentation import get_default_augmentation
from nnunet.training.dataloading.dataset_loading import DataLoader3D, unpack_dataset
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.paths import network_training_output_dir
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from nnunet.utilities.one_hot_encoding import to_one_hot
import shutil

matplotlib.use("agg")


class nnUNetTrainerCascadeFullRes(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, previous_trainer="nnUNetTrainer", fp16=False):
        super(nnUNetTrainerCascadeFullRes, self).__init__(plans_file, fold, output_folder, dataset_directory,
                                                          batch_dice, stage, unpack_data, deterministic, fp16)
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, previous_trainer, fp16)

        if self.output_folder is not None:
            task = self.output_folder.split("/")[-3]
            plans_identifier = self.output_folder.split("/")[-2].split("__")[-1]

            folder_with_segs_prev_stage = join(network_training_output_dir, "3d_lowres",
                                               task, previous_trainer + "__" + plans_identifier, "pred_next_stage")
            if not isdir(folder_with_segs_prev_stage):
                raise RuntimeError(
                    "Cannot run final stage of cascade. Run corresponding 3d_lowres first and predict the "
                    "segmentations for the next stage")
            self.folder_with_segs_from_prev_stage = folder_with_segs_prev_stage
            # Do not put segs_prev_stage into self.output_folder as we need to unpack them for performance and we
            # don't want to do that in self.output_folder because that one is located on some network drive.
            self.folder_with_segs_from_prev_stage_for_train = join(self.dataset_directory, "segs_prev_stage")
        else:
            self.folder_with_segs_from_prev_stage = None
            self.folder_with_segs_from_prev_stage_for_train = None

    def do_split(self):
        super(nnUNetTrainerCascadeFullRes, self).do_split()
        for k in self.dataset:
            self.dataset[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                               k + "_segFromPrevStage.npz")
            assert isfile(self.dataset[k]['seg_from_prev_stage_file']), \
                "seg from prev stage missing: %s" % (self.dataset[k]['seg_from_prev_stage_file'])
        for k in self.dataset_val:
            self.dataset_val[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                   k + "_segFromPrevStage.npz")
        for k in self.dataset_tr:
            self.dataset_tr[k]['seg_from_prev_stage_file'] = join(self.folder_with_segs_from_prev_stage,
                                                                  k + "_segFromPrevStage.npz")

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 True, oversample_foreground_percent=self.oversample_foreground_percent)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, True,
                                  oversample_foreground_percent=self.oversample_foreground_percent)
        else:
            raise NotImplementedError
        return dl_tr, dl_val

    def process_plans(self, plans):
        super(nnUNetTrainerCascadeFullRes, self).process_plans(plans)
        self.num_input_channels += (self.num_classes - 1)  # for seg from prev stage

    def setup_DA_params(self):
        super(nnUNetTrainerCascadeFullRes, self).setup_DA_params()
        self.data_aug_params['selected_seg_channels'] = [0, 1]
        self.data_aug_params['all_segmentation_labels'] = list(range(1, self.num_classes))
        self.data_aug_params['move_last_seg_chanel_to_data'] = True
        self.data_aug_params['advanced_pyramid_augmentations'] = True

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % self.stage)
        if training:
            # copy segs from prev stage to separate folder and extract them

            # If we don't do this then we need to make sure to manually delete the folder if we want to update
            # segs_from_prev_stage. I will probably forget to do so, so I leave this in as a safeguard
            if isdir(self.folder_with_segs_from_prev_stage_for_train):
                shutil.rmtree(self.folder_with_segs_from_prev_stage_for_train)

            maybe_mkdir_p(self.folder_with_segs_from_prev_stage_for_train)
            segs_from_prev_stage_files = subfiles(self.folder_with_segs_from_prev_stage, suffix='.npz')
            for s in segs_from_prev_stage_files:
                shutil.copy(s, self.folder_with_segs_from_prev_stage_for_train)

            # if we don't do this then performance is shit
            if self.unpack_data:
                unpack_dataset(self.folder_with_segs_from_prev_stage_for_train)

            self.folder_with_segs_from_prev_stage = self.folder_with_segs_from_prev_stage_for_train

            self.setup_DA_params()

            if self.folder_with_preprocessed_data is not None:
                self.dl_tr, self.dl_val = self.get_basic_generators()

                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                     self.data_aug_params['patch_size_for_spatialtransform'],
                                                                     self.data_aug_params)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())))
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())))
        else:
            pass
        self.initialize_network_optimizer_and_scheduler()
        assert isinstance(self.network, SegmentationNetwork)
        self.was_initialized = True

    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, validation_folder_name='validation'):
        """

        :param do_mirroring:
        :param use_train_mode:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param use_nifti:
        :param save_softmax:
        :param use_gaussian:
        :param use_temporal_models:
        :return:
        """
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)

        if do_mirroring:
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        process_manager = Pool(2)
        results = []

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            data = np.load(self.dataset[k]['data_file'])['data']

            # concat segmentation of previous step
            seg_from_prev_stage = np.load(join(self.folder_with_segs_from_prev_stage,
                                               k + "_segFromPrevStage.npz"))['data'][None]

            transpose_forward = self.plans.get('transpose_forward')
            if transpose_forward is not None:
                data = data.transpose([0] + [i+1 for i in transpose_forward])
                seg_from_prev_stage = seg_from_prev_stage.transpose([0] + [i+1 for i in transpose_forward])

            print(data.shape)
            data[-1][data[-1] == -1] = 0
            data_for_net = np.concatenate((data[:-1], to_one_hot(seg_from_prev_stage[0], range(1, self.num_classes))))
            softmax_pred = self.predict_preprocessed_data_return_softmax(data_for_net, do_mirroring, 1,
                                                                         use_train_mode, 1, mirror_axes, tiled,
                                                                         True, step, self.patch_size,
                                                                         use_gaussian=use_gaussian)

            if transpose_forward is not None:
                transpose_backward = self.plans.get('transpose_backward')
                softmax_pred = softmax_pred.transpose([0] + [i+1 for i in transpose_backward])

            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            if save_softmax:
                softmax_fname = join(output_folder, fname + ".npz")
            else:
                softmax_fname = None

            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.9): # *0.9 just to be save
                np.save(fname + ".npy", softmax_pred)
                softmax_pred = fname + ".npy"
            results.append(process_manager.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, 1, None, None, None, softmax_fname, None),
                                                          )
                                                         )
                           )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"), json_name=job_name,
                             json_author="Fabian", json_description="",
                             json_task=task)
