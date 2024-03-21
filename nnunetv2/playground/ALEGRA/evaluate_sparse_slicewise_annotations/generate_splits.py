import numpy as np

from nnunetv2.playground.ALEGRA.evaluate_sparse_slicewise_annotations.configuration import *
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import nnunetv2.paths as paths

if __name__ == '__main__':
    np.random.seed(1234)
    for d in datasets:
        dataset_name = maybe_convert_to_dataset_name(d)
        train_identifiers = [i[:-7] for i in nifti_files(join(paths.nnUNet_raw, dataset_name, 'labelsTr'), join=False)]
        split = []

        if d != 980: # shitty fuck fuck making things complicated
            # pick 20% as validation set
            valset = list(np.random.choice(train_identifiers, round(len(train_identifiers) * 0.2), replace=False))
        else:
            # find mice identifiers
            mice_identifiers = np.unique([i.split('__')[0] for i in train_identifiers])
            val_mice = list(np.random.choice(list(mice_identifiers), round(len(mice_identifiers) * 0.2), replace=False))
            valset = [i for i in train_identifiers if any([i.startswith(j) for j in val_mice])]

        trainset = [i for i in train_identifiers if i not in valset]

        for pa in percent_of_cases_annotated:
            num_cases = max(1, round(len(trainset) * pa / 100))
            for s in range(num_runs):
                tr_cases = list(np.random.choice(trainset, num_cases, replace=False))
                split.append({'train': tr_cases, 'val': valset})
        save_json(split, join(paths.nnUNet_preprocessed, dataset_name, 'splits_final.json'))
