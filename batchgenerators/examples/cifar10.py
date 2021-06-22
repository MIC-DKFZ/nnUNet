import re
import torch
from batchgenerators.dataloading import MultiThreadedAugmenter
import numpy as np
import os
from batchgenerators.dataloading.data_loader import DataLoaderFromDataset
from batchgenerators.datasets.cifar import HighPerformanceCIFARLoader, CifarDataset
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms import NumpyToTensor, Compose
from torch._six import int_classes, string_classes, container_abcs
from torch.utils.data.dataloader import numpy_type_map

_use_shared_memory = False


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


if __name__ == '__main__':
    ### current implementation of betchgenerators stuff for this script does not use _use_shared_memory!

    from time import time
    batch_size = 50
    num_workers = 8
    pin_memory = False
    num_epochs = 3
    dataset_dir = '/media/fabian/data/data/cifar10'
    numpy_to_tensor = NumpyToTensor(['data', 'labels'], cast_to=None)
    fname = os.path.join(dataset_dir, 'cifar10_training_data.npz')
    dataset = np.load(fname)
    cifar_dataset_as_arrays = (dataset['data'], dataset['labels'], dataset['filenames'])
    print('batch_size', batch_size)
    print('num_workers', num_workers)
    print('pin_memory', pin_memory)
    print('num_epochs', num_epochs)

    tr_transforms = [SpatialTransform((32, 32))] * 1  # SpatialTransform is computationally expensive and we need some
    # load on CPU so we just stack 5 of them on top of each other
    tr_transforms.append(numpy_to_tensor)
    tr_transforms = Compose(tr_transforms)

    cifar_dataset = CifarDataset(dataset_dir, train=True, transform=tr_transforms)

    dl = DataLoaderFromDataset(cifar_dataset, batch_size, num_workers, 1)
    mt = MultiThreadedAugmenter(dl, None, num_workers, 1, None, pin_memory)

    batches = 0
    for _ in mt:
        batches += 1
    assert len(_['data'].shape) == 4

    assert batches == len(cifar_dataset) / batch_size  # this assertion only holds if len(datset) is divisible by
    # batch size

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in mt:
            batches += 1
    stop = time()
    print('batchgenerators took %03.4f seconds' % (stop - start))

    # The best I can do:

    dl = HighPerformanceCIFARLoader(cifar_dataset_as_arrays, batch_size, num_workers, 1) # this circumvents the
    # default_collate function, just to see if that is slowing things down
    mt = MultiThreadedAugmenter(dl, tr_transforms, num_workers, 1, None, pin_memory)

    batches = 0
    for _ in mt:
        batches += 1
    assert len(_['data'].shape) == 4

    assert batches == len(cifar_dataset_as_arrays[0]) / batch_size  # this assertion only holds if len(datset) is
    # divisible by batch size

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in mt:
            batches += 1
    stop = time()
    print('high performance batchgenerators %03.4f seconds' % (stop - start))


    from torch.utils.data import DataLoader as TorchDataLoader

    trainloader = TorchDataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=default_collate)

    batches = 0
    for _ in iter(trainloader):
        batches += 1
    assert len(_['data'].shape) == 4

    start = time()
    for _ in range(num_epochs):
        batches = 0
        for _ in trainloader:
            batches += 1
    stop = time()
    print('pytorch took %03.4f seconds' % (stop - start))
