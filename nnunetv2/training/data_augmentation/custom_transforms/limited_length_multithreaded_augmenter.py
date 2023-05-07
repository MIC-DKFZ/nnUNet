from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    def __init__(self, my_imaginary_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.len = my_imaginary_length

    def __len__(self):
        return self.len

# class LimitedLenWrapper(MultiThreadedAugmenter):
#     def __init__(self, my_imaginary_length, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.len = my_imaginary_length

#     def __len__(self):
#         return self.len
