import os
import shutil
import tarfile
from urllib.request import urlretrieve

import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.dataset import Dataset
from batchgenerators.utilities.file_and_folder_operations import join


def unpickle(file):
    '''
    taken from http://www.cs.toronto.edu/~kriz/cifar.html
    :param file:
    :return:
    '''
    import pickle

    with open(file, 'rb') as fo:
        dc = pickle.load(fo, encoding='bytes')
    return dc


def maybe_download_and_prepare_cifar(target_dir, cifar=10):
    '''
    Checks if cifar is already present in target_dir and downloads it if not.
    CIFAR comes in 5 batches that need to be unpickled. What a mess.
    We stack all 5 batches together to one single npy array. No idea why they are being so complicated
    :param target_dir:
    :return:
    '''
    if not os.path.isfile(os.path.join(target_dir, 'cifar%d_test_data.npz' % cifar)) or not \
            os.path.isfile(os.path.join(target_dir, 'cifar%d_training_data.npz' % cifar)):
        print('downloading CIFAR%d...' % cifar)
        urlretrieve('http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % cifar, join(target_dir, 'cifar-%d-python.tar.gz' % cifar))

        tar = tarfile.open(os.path.join(target_dir, 'cifar-%d-python.tar.gz' % cifar), "r:gz")
        tar.extractall(path=target_dir)
        tar.close()

        data = []
        labels = []
        filenames = []

        for batch in range(1, 6):
            loaded = unpickle(os.path.join(target_dir, 'cifar-%d-batches-py' % cifar, 'data_batch_%d' % batch))
            data.append(loaded[b'data'].reshape((loaded[b'data'].shape[0], 3, 32, 32)).astype(np.uint8))
            labels += [int(i) for i in loaded[b'labels']]
            filenames += [str(i) for i in loaded[b'filenames']]

        data = np.vstack(data)
        labels = np.array(labels)
        filenames = np.array(filenames)

        np.savez_compressed(os.path.join(target_dir, 'cifar%d_training_data.npz' % cifar), data=data, labels=labels,
                            filenames=filenames)

        test = unpickle(os.path.join(target_dir, 'cifar-%d-batches-py' % cifar, 'test_batch'))
        data = test[b'data'].reshape((test[b'data'].shape[0], 3, 32, 32)).astype(np.uint8)
        labels = [int(i) for i in test[b'labels']]
        filenames = [i for i in test[b'filenames']]

        np.savez_compressed(os.path.join(target_dir, 'cifar%d_test_data.npz' % cifar), data=data, labels=labels,
                            filenames=filenames)

        # clean up
        shutil.rmtree(os.path.join(target_dir, 'cifar-%d-batches-py' % cifar))
        os.remove(os.path.join(target_dir, 'cifar-%d-python.tar.gz' % cifar))


class CifarDataset(Dataset):
    def __init__(self, dataset_directory, train=True, transform=None, cifar=10):
        super(CifarDataset, self).__init__()
        self.transform = transform
        maybe_download_and_prepare_cifar(dataset_directory)

        self.train = train

        # load appropriate data
        if train:
            fname = os.path.join(dataset_directory, 'cifar%d_training_data.npz' % cifar)
        else:
            fname = os.path.join(dataset_directory, 'cifar%d_test_data.npz' % cifar)

        dataset = np.load(fname)

        self.data = dataset['data']
        self.labels = dataset['labels']
        self.filenames = dataset['filenames']

    def __getitem__(self, item):
        data_dict = {'data': self.data[item:item+1].astype(np.float32), 'labels': self.labels[item], 'filenames': self.filenames[item]}
        if self.transform is not None:
            data_dict = self.transform(**data_dict)
        return data_dict

    def __len__(self):
        return len(self.data)


class HighPerformanceCIFARLoader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1, infinite=False,
                 return_incomplete=False):
        super(HighPerformanceCIFARLoader, self).__init__(data, batch_size, num_threads_in_multithreaded,
                                                         seed_for_shuffle, infinite=infinite,
                                                         return_incomplete=return_incomplete)
        self.indices = np.arange(len(data[0]))

    def generate_train_batch(self):
        indices = self.get_indices()

        data = self._data[0][indices]
        labels = self._data[1][indices]
        filenames = self._data[2][indices]

        return {'data': data.astype(np.float32), 'labels': labels, 'filenames': filenames}

