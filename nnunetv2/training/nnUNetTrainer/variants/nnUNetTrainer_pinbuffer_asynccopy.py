import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.pinned_buffer_nondet_mta import PinnedBufferNonDetMultiThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast


class CUDAPrefetcher:
    """
    Pulls (cpu_batch, handle) from a pinned-buffer augmenter and asynchronously copies to GPU
    on a dedicated stream. Exposes an iterator yielding gpu_batch dicts.
    """

    def __init__(self, loader, device, release_fn, wait_time_safety=False):
        self.loader = loader
        self.device = device
        self.release_fn = release_fn  # function(handle) -> None

        assert device.type == "cuda", "CUDAPrefetcher is only for CUDA"
        self.copy_stream = torch.cuda.Stream(device=device)

        self._next_gpu_batch = None
        self._next_event = None
        self._next_handle = None

    def _to_device_async(self, batch: dict):
        # Copy only what needs to go to GPU. Assumes stable keys and that CPU tensors are pinned.
        data = batch["data"]
        target = batch["target"]

        with torch.cuda.stream(self.copy_stream):
            data_gpu = data.to(self.device, non_blocking=True)
            if isinstance(target, list):
                target_gpu = [t.to(self.device, non_blocking=True) for t in target]
            else:
                target_gpu = target.to(self.device, non_blocking=True)

            gpu_batch = dict(batch)
            gpu_batch["data"] = data_gpu
            gpu_batch["target"] = target_gpu

            ev = torch.cuda.Event()
            ev.record(self.copy_stream)

        return gpu_batch, ev

    def _prefetch(self):
        cpu_batch, handle = next(self.loader)
        gpu_batch, ev = self._to_device_async(cpu_batch)
        self._next_gpu_batch = gpu_batch
        self._next_event = ev
        self._next_handle = handle

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_gpu_batch is None:
            self._prefetch()

        # Promote prefetched batch to "current"
        gpu_batch = self._next_gpu_batch
        ev = self._next_event
        handle = self._next_handle

        # Make default stream wait for the H2D copies to complete
        torch.cuda.current_stream(self.device).wait_event(ev)

        # Now it is safe to reuse pinned buffers for this batch
        if handle is not None and getattr(handle, "slot_id", -1) >= 0:
            self.release_fn(handle)

        # Clear current before starting next prefetch (reduces peak live objects)
        self._next_gpu_batch = None
        self._next_event = None
        self._next_handle = None

        # Prefetch the following batch after we made the current ready
        self._prefetch()

        return gpu_batch

    def close(self):
        if self._next_handle is not None and getattr(self._next_handle, "slot_id", -1) >= 0:
            self.release_fn(self._next_handle)
            self._next_handle = None
            self._next_gpu_batch = None
            self._next_event = None

class nnUNetTrainer_pinbuffer_asynccopy(nnUNetTrainer):
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    def run_training(self):
        self.on_train_start()

        if self.device.type == "cuda":
            train_iter = CUDAPrefetcher(
                self.dataloader_train,
                self.device,
                release_fn=self.dataloader_train.release
            )
        else:
            train_iter = iter(self.dataloader_train)

        if self.device.type == "cuda":
            val_iter = CUDAPrefetcher(
                self.dataloader_val,
                self.device,
                release_fn=self.dataloader_val.release
            )
        else:
            val_iter = iter(self.dataloader_val)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []


            for _ in range(self.num_iterations_per_epoch):
                batch = next(train_iter)
                train_outputs.append(self.train_step(batch))

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []

                for _ in range(self.num_val_iterations_per_epoch):
                    batch = next(val_iter)
                    val_outputs.append(self.validation_step(batch))

                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        if self.device.type == "cuda":
            train_iter.close()
            val_iter.close()

        self.on_train_end()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = PinnedBufferNonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=6,
                                                                    pinned_pool_size=3,
                                                                    seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.02,
                                                                    liveness_check_interval=5)
            mt_gen_val = PinnedBufferNonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                                  pinned_pool_size=3,
                                                      num_cached=6, seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.02,
                                                                  liveness_check_interval=5)
        # # let's get this party started
        batch, handle = next(mt_gen_train)
        mt_gen_train.release(handle)

        batch, handle = next(mt_gen_val)
        mt_gen_val.release(handle)
        return mt_gen_train, mt_gen_val