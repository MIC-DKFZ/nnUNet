from time import time

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.examples.brats2017.brats2017_dataloader_3D import get_list_of_patients, BraTS2017DataLoader3D, \
    get_train_transform
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.utilities.data_splitting import get_split_deterministic


class BraTS2017DataLoader2D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234, return_incomplete=False,
                 shuffle=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         True)
        self.patch_size = patch_size
        self.num_modalities = 4
        self.indices = list(range(len(data)))

    @staticmethod
    def load_patient(patient):
        return BraTS2017DataLoader3D.load_patient(patient)

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)

            # patient data is a memmap. If we extract just one slice then just this one slice will be read from the
            # disk, so no worries!
            slice_idx = np.random.choice(patient_data.shape[1])
            patient_data = patient_data[:, slice_idx]

            # this will only pad patient_data if its shape is smaller than self.patch_size
            patient_data = pad_nd_image(patient_data, self.patch_size)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type="random")

            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

            metadata.append(patient_metadata)
            patient_names.append(j)

        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}


if __name__ == "__main__":
    patients = get_list_of_patients(brats_preprocessed_folder)

    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)

    patch_size = (160, 160)
    batch_size = 48

    # I recommend you don't use 'iteration oder all training data' as epoch because in patch based training this is
    # really not super well defined. If you leave all arguments as default then each batch sill contain randomly
    # selected patients. Since we don't care about epochs here we can set num_threads_in_multithreaded to anything.
    dataloader = BraTS2017DataLoader2D(train, batch_size, patch_size, 1)

    batch = next(dataloader)
    try:
        from batchviewer import view_batch
        # batch viewer can show up to 4d tensors. We can show only one sample, but that should be sufficient here
        view_batch(np.concatenate((batch['data'][0], batch['seg'][0]), 0)[:, None])
    except ImportError:
        view_batch = None
        print("you can visualize batches with batchviewer. It's a nice and handy tool. You can get it here: "
              "https://github.com/FabianIsensee/BatchViewer")

    # now we have some DataLoader. Let's go an get some augmentations

    # first let's collect all shapes, you will see why later
    shapes = [BraTS2017DataLoader2D.load_patient(i)[0].shape[2:] for i in patients]
    max_shape = np.max(shapes, 0)
    max_shape = np.max((max_shape, patch_size), 0)

    # we create a new instance of DataLoader. This one will return batches of shape max_shape. Cropping/padding is
    # now done by SpatialTransform. If we do it this way we avoid border artifacts (the entire brain of all cases will
    # be in the batch and SpatialTransform will use zeros which is exactly what we have outside the brain)
    # this is viable here but not viable if you work with different data. If you work for example with CT scans that
    # can be up to 500x500x500 voxels large then you should do this differently. There, instead of using max_shape you
    # should estimate what shape you need to extract so that subsequent SpatialTransform does not introduce border
    # artifacts
    dataloader_train = BraTS2017DataLoader2D(train, batch_size, max_shape, 1)

    # during training I like to run a validation from time to time to see where I am standing. This is not a correct
    # validation because just like training this is patch-based but it's good enough. We don't do augmentation for the
    # validation, so patch_size is used as shape target here
    dataloader_validation = BraTS2017DataLoader2D(val, batch_size, patch_size, 1)

    tr_transforms = get_train_transform(patch_size)

    # finally we can create multithreaded transforms that we can actually use for training
    # we don't pin memory here because this is pytorch specific.
    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=num_threads_for_brats_example,
                                    num_cached_per_queue=3,
                                    seeds=None, pin_memory=False)
    # we need less processes for vlaidation because we dont apply transformations
    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=max(1, num_threads_for_brats_example // 2), num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)

    # lets start the MultiThreadedAugmenter. This is not necessary but allows them to start generating training
    # batches while other things run in the main thread
    tr_gen.restart()
    val_gen.restart()

    # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
    # inifinite examples! Don't do "for batch in tr_gen:"!!!):
    num_batches_per_epoch = 10
    num_validation_batches_per_epoch = 3
    num_epochs = 5
    # let's run this to get a time on how long it takes
    time_per_epoch = []
    start = time()
    for epoch in range(num_epochs):
        start_epoch = time()
        for b in range(num_batches_per_epoch):
            batch = next(tr_gen)
            # do network training here with this batch

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)
            # run validation here
        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)
    end = time()
    total_time = end - start
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (num_epochs, total_time, str(time_per_epoch)))

    # if you notice that you have CPU usage issues, reduce the probability with which the spatial transformations are
    # applied in get_train_transform (down to 0.1 for example). SpatialTransform is the most expensive transform

    # if you wish to visualize some augmented examples, install batchviewer and uncomment this
    if view_batch is not None:
        for _ in range(4):
            batch = next(tr_gen)
            view_batch(np.concatenate((batch['data'][0], batch['seg'][0]), 0)[:, None])
    else:
        print("Cannot visualize batches, install batchviewer first")
