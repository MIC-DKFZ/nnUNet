from batchgenerators.dataloading import MultiThreadedAugmenter, SlimDataLoaderBase
import numpy as np


class DummyDL(SlimDataLoaderBase):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDL, self).__init__(None, None, num_threads_in_mt)
        self._data = list(range(100))
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[idx]
        else:
            self.reset()
            raise StopIteration


class DummyDLWithShuffle(DummyDL):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDLWithShuffle, self).__init__(num_threads_in_mt)
        self.num_restarted = 0
        self.data_order = np.arange(len(self._data))

    def reset(self):
        super(DummyDLWithShuffle, self).reset()
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[self.data_order[idx]]
        else:
            self.reset()
            raise StopIteration


if __name__ == "__main__":
    """
    Why is is so hard to iterate only once over my entire training dataset when MultiThreadedAugmenter is used?
    This is because MultiThreadedAugmenter will spawn num_threads workers and each worker will hold a copy of the entire
    pipeline, including the DataLoader. Therefore, if your DataLoader is configured to run over the training data once, but 
    you have 8 threads then what you will be getting from the MultiThreadedAugmenter is an iteration over eight times your 
    training dataset"""

    """
    HELP I want to iterate over all my training data once per epoch.
    Say no more. We go your back. Here is a simple example how you can do that.

    We create a dummy dataloader that has the numbers of 0 to 99 in its _data variable. In the MultiThreadedAugmenter, each 
    DataLoader will know what thread ID it has. We use that information to iterate over the training data. Since there are 
    3 threads, each individual dataloader must return every third item (and start in a different position)
    """

    dl = DummyDL(num_threads_in_mt=3)
    mt = MultiThreadedAugmenter(dl, None, 3, 1, None)

    for i in mt:
        print(i)


    """
    You can run the mt as often as you want because the DataLoader it will reset itself before raising StopIteration
    """
    for i in mt:
        print(i)

    for i in mt:
        print(i)


    """
    But wait. Isn't it suboptimal to iterate over training data always in the same order? Correct. Try this:
    """

    dl = DummyDLWithShuffle(num_threads_in_mt=3)
    mt = MultiThreadedAugmenter(dl, None, 3, 1, None)

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert len(np.unique(batches)) == 100 and len(batches) == 100 # assert makes sure we got what we wanted

    """
    Once again you can run that as often as you want
    """

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert len(np.unique(batches)) == 100 and len(batches) == 100

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert len(np.unique(batches)) == 100 and len(batches) == 100
