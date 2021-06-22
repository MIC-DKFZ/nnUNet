from abc import ABCMeta, abstractmethod


class Dataset(object):
    def __init__(self):
        __metaclass__ = ABCMeta

    @abstractmethod
    def __getitem__(self, item):
        '''
        needs to return a data_dict for the sample at the position item
        :param item:
        :return:
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''
        returns how many items the dataset has
        :return:
        '''
        pass



