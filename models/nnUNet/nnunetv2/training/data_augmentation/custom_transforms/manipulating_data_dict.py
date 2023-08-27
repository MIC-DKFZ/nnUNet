from batchgenerators.transforms.abstract_transforms import AbstractTransform


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove: str):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict
