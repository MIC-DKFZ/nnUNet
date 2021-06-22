# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABCMeta, abstractmethod
from builtins import object
import warnings
from collections import OrderedDict
from warnings import warn
import numpy as np

from batchgenerators.dataloading.dataset import Dataset


class DataLoaderBase(object):
    """ Derive from this class and override generate_train_batch. If you don't want to use this you can use any
    generator.
    You can modify this class however you want. How the data is presented as batch is you responsibility. You can sample
    randomly, cycle through the training examples or sample the dtaa according to a specific pattern. Just make sure to
    use our default data structure!
    {'data':your_batch_of_shape_(b, c, x, y(, z)),
    'seg':your_batch_of_shape_(b, c, x, y(, z)),
    'anything_else1':whatever,
    'anything_else2':whatever2,
    ...}

    (seg is optional)

    Args:
        data (anything): Your dataset. Stored as member variable self._data

        BATCH_SIZE (int): batch size. Stored as member variable self.BATCH_SIZE

        num_batches (int): How many batches will be generated before raising StopIteration. None=unlimited. Careful
        when using MultiThreadedAugmenter: Each process will produce num_batches batches.

        seed (False, None, int): seed to seed the numpy rng with. False = no seeding

    """
    def __init__(self, data, BATCH_SIZE, num_batches=None, seed=False):
        warnings.simplefilter("once", DeprecationWarning)
        warn("This DataLoader will soon be removed. Migrate everything to SlimDataLoaderBase now!", DeprecationWarning)
        __metaclass__ = ABCMeta
        self._data = data
        self.BATCH_SIZE = BATCH_SIZE
        if num_batches is not None:
            warn("We currently strongly discourage using num_batches != None! That does not seem to work properly")
        self._num_batches = num_batches
        self._seed = seed
        self._was_initialized = False
        if self._num_batches is None:
            self._num_batches = 1e100
        self._batches_generated = 0
        self.thread_id = 0

    def reset(self):
        if self._seed is not False:
            np.random.seed(self._seed)
        self._was_initialized = True
        self._batches_generated = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        if not self._was_initialized:
            self.reset()
        if self._batches_generated >= self._num_batches:
            self._was_initialized = False
            raise StopIteration
        minibatch = self.generate_train_batch()
        self._batches_generated += 1
        return minibatch

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.

        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()

        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!

        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, batch_size, num_threads_in_multithreaded=1, seed_for_shuffle=None, return_incomplete=False,
                 shuffle=True, infinite=False):
        """

        :param data: will be stored in self._data for use in generate_train_batch
        :param batch_size: will be used by get_indices to return the correct number of indices
        :param num_threads_in_multithreaded: num_threads_in_multithreaded necessary for synchronization of dataloaders
        when using multithreaded augmenter
        :param seed_for_shuffle: for reproducibility
        :param return_incomplete: whether or not to return batches that are incomplete. Only applies is infinite=False.
        If your data has len of 34 and your batch size is 32 then there return_incomplete=False will make this loader
        return only onebatch of shapre 32 (omitting 2 of your training examples). If return_incomplete=True a second
        batch with batch size 2 will be returned.
        :param shuffle: if True, the order of the indices will be shuffled between epochs. Only applies if infinite=False
        :param infinite: if True, each batch contains randomly (uniformly) sampled indices. An unlimited number of
        batches is returned. If False, DataLoader will iterate over the data only once
        """
        super(DataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded)
        self.infinite = infinite
        self.shuffle = shuffle
        self.return_incomplete = return_incomplete
        self.seed_for_shuffle = seed_for_shuffle
        self.rs = np.random.RandomState(self.seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.last_reached = False

        # when you derive, make sure to set this! We can't set it here because we don't know what data will be like
        self.indices = None

    def reset(self):
        assert self.indices is not None

        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        if not self.infinite and self.shuffle:
            self.rs.shuffle(self.indices)

        self.last_reached = False

    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=None)

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    @abstractmethod
    def generate_train_batch(self):
        '''
        make use of self.get_indices() to know what indices to work on!
        :return:
        '''
        pass


def default_collate(batch):
    '''
    heavily inspired by the default_collate function of pytorch
    :param batch:
    :return:
    '''
    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    else:
        raise TypeError('unknown type for batch:', type(batch))


class DataLoaderFromDataset(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        '''
        A simple dataloader that can take a Dataset as data.
        It is not super efficient because I cannot make too many hard assumptions about what data_dict will contain.
        If you know what you need, implement your own!
        :param data:
        :param batch_size:
        :param num_threads_in_multithreaded:
        :param seed_for_shuffle:
        '''
        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        assert isinstance(self._data, Dataset)
        self.indices = np.arange(len(data))

    def generate_train_batch(self):
        indices = self.get_indices()

        batch = [self._data[i] for i in indices]

        return self.collate_fn(batch)
