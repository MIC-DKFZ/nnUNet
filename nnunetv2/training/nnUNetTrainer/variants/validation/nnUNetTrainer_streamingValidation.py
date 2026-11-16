"""
Streaming final validation, and the trainers that use it.

nnUNetTrainer.perform_actual_validation predicts the (num_heads, *internal_shape) logits of each validation case and
hands them to export workers, which resample them to (num_heads, *original_shape) on the CPU before the argmax. With
100+ classes on large images that resampled volume is tens of GB per case, once per export worker in flight, and the
final validation runs out of RAM. StreamingValidationMixin replaces it with nnUNetPredictorStreaming
(nnunetv2/inference/predict_streaming.py): prediction, resampling to the original geometry, argmax and uncropping run in
one pass over a slab of the prediction on the GPU, and the export workers only write the finished segmentation.
Training is unchanged; only the final validation (also --val) takes this path.

Resampling follows the plans, as nnUNetv2_predict_streaming does: torch trilinear, or the plans' separate-z path for
the cases where it fires, decided per case exactly as the stock export decides it. Resampling that streaming cannot
reproduce (e.g. cubic) is replaced by torch trilinear, and the training log says so. Otherwise the segmentations match
those of the stock validation up to float rounding at argmax boundaries.

Feature ring: the network's 1x1x1 output layer is detached for the validation, the slab buffer holds the decoder's last
features instead of the class logits, and the layer is applied per output slice (module docstring of predict_streaming).
Same segmentation up to float rounding, several GB less VRAM with many classes. The layer goes back into the trainer's
network when the validation ends, also on an exception. streaming_feature_ring = False on the trainer class keeps the
logits.

Scope: 3d configurations. No cascade, no next-stage prediction and no probability export: they need the full
probability volume, which this validation never builds (AssertionError).

The trainers at the end of this file:
  * nnUNetTrainer_SV: nnUNetTrainer with streaming validation.
  * nnUNetTrainer_2000epochs_NoMirroring_SV: 2000 epochs without mirroring, stock augmentation.
  * nnUNetTrainer_DASegOrd0_NoMirroring_SV and its 2000 / 5000 epoch versions: a recipe for datasets with many classes.
    nnUNetTrainer_DASegOrd0_NoMirroring resamples the segmentation with order 0 in the spatial augmentation (the
    default interpolates it as one channel per class, which makes the data augmentation CPU bound at 100+ classes) and
    does not mirror. ChunkedLossMixin chunks CE and Dice over the batch once batch * classes * voxels reaches the int32
    indexing limit of the CE backward; below that limit it computes exactly the stock loss.
"""
import multiprocessing
import warnings
from time import sleep

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import distributed as dist

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.predict_streaming import nnUNetPredictorStreaming, separate_z_rule
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.streaming_pipeline import write_segmentation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import \
    nnUNetTrainer_DASegOrd0_NoMirroring
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_chunkedLoss import ChunkedLossMixin
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import \
    nnUNetTrainer_2000epochs_NoMirroring
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context, empty_cache, is_optimized_module, \
    set_default_inductor_cache_dir, unwrap_network


class nnUNetPredictorStreamingFromTrainer(nnUNetPredictorStreaming):
    """
    nnUNetPredictorStreaming fed by nnUNetTrainer.manual_initialization instead of by a results folder.

    The trainer hands over the live network with the weights already in it, so there is nothing to ensemble and no
    state_dict to swap: it passes list_of_parameters = [None] (the streaming loop uses its length as the fold count)
    and the per-slab weight load is a no-op. This also avoids load_state_dict on a torch.compile'd or DDP wrapped
    module during validation.
    """

    def _load_fold(self, fold_index: int):
        assert fold_index == 0, 'the trainer only ever has one set of weights loaded'

    def setup_feature_ring(self):
        """
        Detach the output layer of the trainer's network (detach_head; heads from the live weights, since
        list_of_parameters is [None]) and run the headless plain module from here on - compiled again if the trainer's
        network was compiled, without the DDP wrapper (forward only). restore_head() puts the layer back.
        """
        net = self.network
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            net = net.module
        compiled = is_optimized_module(net)
        if self.detach_head():
            raw = unwrap_network(self.network)
            if compiled:
                set_default_inductor_cache_dir()
                raw = torch.compile(raw)
            self.network = raw

    def predict_preprocessed_case(self, data: torch.Tensor, properties: dict) -> np.ndarray:
        """
        An already preprocessed case (as it sits in nnUNet_preprocessed) -> the finished segmentation in the original
        image geometry, resampling, argmax and uncropping included. Only a file write is left to do.
        """
        return self._predict_and_export_streaming(data, properties)


class StreamingValidationMixin:
    """
    Replaces nnUNetTrainer.perform_actual_validation with the streaming one (module docstring). Mix in BEFORE the
    trainer, e.g. ``class MyTrainer_SV(StreamingValidationMixin, MyTrainer)``. Changes nothing about training.
    """
    # accumulate the decoder's last features instead of the class logits (module docstring); False = logits
    streaming_feature_ring = True

    def initialize(self):
        # refuse a configuration the final validation cannot handle before training, not after it
        self._streaming_validation_sanity_checks()
        super().initialize()

    def _streaming_validation_sanity_checks(self):
        assert not self.is_cascaded, \
            ('Streaming validation does not support the cascade: it never materializes the full probability volume, '
             'so there is nothing to feed a next stage. Validate with the stock nnUNetTrainer.')
        assert self.configuration_manager.next_stage_names is None, \
            (f'Configuration {self.configuration_name} has next stages {self.configuration_manager.next_stage_names}; '
             f'predicting them requires the full probability volume, which streaming validation does not build. '
             f'Validate with the stock nnUNetTrainer.')
        assert len(self.configuration_manager.patch_size) == 3, \
            (f'Streaming validation supports 3d configurations only, got patch size '
             f'{self.configuration_manager.patch_size}.')

    def perform_actual_validation(self, save_probabilities: bool = False):
        assert not save_probabilities, \
            ('Streaming validation cannot export probabilities: the full probability volume is never formed. '
             'Use the stock nnUNetTrainer if you need them.')
        self._streaming_validation_sanity_checks()

        self.set_deep_supervision_enabled(False)
        self.network.eval()
        # unlike the stock validation, the work happens in VRAM (the slab buffer). Give the allocator a clean slate
        # after training so that the validation does not fail on fragmentation.
        empty_cache(self.device)

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetPredictorStreamingFromTrainer(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                                        perform_everything_on_device=True, device=self.device,
                                                        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
                                                        feature_ring=self.streaming_feature_ring)
        # [None]: one resident model, whose weights the network already holds (nnUNetPredictorStreamingFromTrainer)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, [None],
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        # feature ring: the output layer comes off the trainer's network here and goes back in the finally below
        predictor.setup_feature_ring()
        self._log_streaming_validation_notice(predictor)

        reader_writer_class = self.plans_manager.image_reader_writer_class
        file_ending = self.dataset_json['file_ending']

        with self.network.no_sync() if self.is_ddp else dummy_context():
            try:
                with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
                    worker_list = [i for i in segmentation_export_pool._pool]
                    validation_output_folder = join(self.output_folder, 'validation')
                    maybe_mkdir_p(validation_output_folder)

                    # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to
                    # distribute the validation keys across the workers.
                    _, val_keys = self.do_split()
                    if self.is_ddp:
                        val_keys = val_keys[self.global_rank:: self.world_size]

                    dataset_val = self.dataset_class(
                        self.preprocessed_dataset_folder, val_keys,
                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

                    results = []

                    for i, k in enumerate(dataset_val.identifiers):
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                   allowed_num_queued=2)
                        while not proceed:
                            sleep(0.1)
                            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                       allowed_num_queued=2)

                        self.print_to_log_file(f"predicting {k}")
                        data, _, _ = dataset_val.load_case(k)
                        properties = dataset_val.get_properties(k)

                        # we do [:] to convert blosc2 to numpy
                        data = data[:]

                        with warnings.catch_warnings():
                            # ignore 'The given NumPy array is not writable' warning
                            warnings.simplefilter("ignore")
                            data = torch.from_numpy(data)
                        # the streaming loop cuts the patches out of this tensor
                        data = data.float()

                        self.print_to_log_file(f'{k}, shape {data.shape}, global rank {self.global_rank}')
                        output_filename_truncated = join(validation_output_folder, k)

                        # prediction, resampling to the original geometry, argmax and uncropping in one streaming
                        # pass. What comes back is the finished segmentation in the original image geometry
                        segmentation = predictor.predict_preprocessed_case(data, properties)
                        del data

                        # the export worker only has to write it (no resampling, no argmax)
                        results.append(
                            segmentation_export_pool.starmap_async(
                                write_segmentation, (
                                    (segmentation, properties, output_filename_truncated, reader_writer_class,
                                     file_ending),
                                )
                            )
                        )
                        del segmentation

                    _ = [r.get() for r in results]
            finally:
                predictor.restore_head()                 # the trainer's network gets its output layer back

            # every rank only waited for its OWN exports above. rank 0 evaluates the whole folder, so it must not
            # start before all other ranks are done writing into it - compute_metrics_on_folder is called with
            # chill=True and would silently evaluate whatever subset happens to be on disk.
            if self.is_ddp:
                dist.barrier()

            if self.global_rank == 0:
                # metrics on rank 0 only, with the CPUs of this node's idle ranks (local_world_size, not world_size)
                metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                    validation_output_folder,
                                                    join(validation_output_folder, 'summary.json'),
                                                    self.plans_manager.image_reader_writer_class(),
                                                    self.dataset_json["file_ending"],
                                                    self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                    self.label_manager.foreground_labels,
                                                    self.label_manager.ignore_label, chill=True,
                                                    num_processes=default_num_processes * self.local_world_size)
                for label in metrics["mean"]:
                    self.logger.log_summary(f"final_val/class_{label}_dice", metrics["mean"][label]["Dice"])
                self.logger.log_summary("final_val/foreground_dice", metrics['foreground_mean']["Dice"])
                self.print_to_log_file("Validation complete", also_print_to_console=True)
                self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                       also_print_to_console=True)

            self.set_deep_supervision_enabled(True)
            compute_gaussian.cache_clear()

        # deliberately NO barrier here, as in nnUNetTrainer.perform_actual_validation: the other ranks are done and
        # have nothing left to synchronize on, so parking them in a collective for the duration of rank 0's metric
        # computation would only expose us to the NCCL watchdog timeout.

    def _log_streaming_validation_notice(self, predictor: nnUNetPredictorStreaming):
        """Say in the training log that this run did not use the stock validation, and how it resamples."""
        cfg = self.configuration_manager.configuration
        fn = cfg.get('resampling_fn_probabilities')
        kwargs = cfg.get('resampling_fn_probabilities_kwargs')
        reproducible, _ = separate_z_rule(self.configuration_manager.resampling_fn_probabilities)

        self.print_to_log_file(
            'STREAMING VALIDATION. Prediction, resampling, argmax and export are fused into one pass over a slab of '
            'the prediction; the full '
            f'({self.label_manager.num_segmentation_heads}, *original_shape) probability volume is never built.',
            also_print_to_console=True)
        if reproducible:
            self.print_to_log_file(
                f'  Resampling follows the plans: resampling_fn_probabilities={fn}, kwargs {kwargs} (torch trilinear, '
                f'or the separate-z path for the cases where it fires, decided per case as in the stock validation). '
                f'Segmentations match the stock validation up to float rounding at argmax boundaries.',
                also_print_to_console=True)
        else:
            self.print_to_log_file(
                f'  WARNING: streaming cannot reproduce resampling_fn_probabilities={fn}, kwargs {kwargs}, and '
                f'resamples the predicted logits with plain torch trilinear (align_corners=False) instead. Dice from '
                f'this run is not strictly comparable to that of the stock validation.',
                also_print_to_console=True)
        self.print_to_log_file(f'  Slab buffer: feature ring {predictor.feature_ring_status}',
                               also_print_to_console=True)


# ----------------------------------------------------------------------------------------------------------------------
# trainers
# ----------------------------------------------------------------------------------------------------------------------
class nnUNetTrainer_SV(StreamingValidationMixin, nnUNetTrainer):
    """nnUNetTrainer with streaming final validation."""


class nnUNetTrainer_2000epochs_NoMirroring_SV(StreamingValidationMixin, nnUNetTrainer_2000epochs_NoMirroring):
    """2000 epochs, no mirroring in augmentation and test-time augmentation, stock augmentation otherwise; streaming
    final validation."""


class nnUNetTrainer_DASegOrd0_NoMirroring_SV(StreamingValidationMixin, ChunkedLossMixin,
                                             nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    Many classes, 1000 epochs: segmentations resampled with order 0 in the spatial augmentation, no mirroring,
    batch-chunked loss (exactly the stock loss below the int32 limit), streaming final validation.
    """


class nnUNetTrainer_DASegOrd0_NoMirroring_2000epochs_SV(nnUNetTrainer_DASegOrd0_NoMirroring_SV):
    """nnUNetTrainer_DASegOrd0_NoMirroring_SV, 2000 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000


class nnUNetTrainer_DASegOrd0_NoMirroring_5000epochs_SV(nnUNetTrainer_DASegOrd0_NoMirroring_SV):
    """nnUNetTrainer_DASegOrd0_NoMirroring_SV, 5000 epochs."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5000
