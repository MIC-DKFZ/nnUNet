import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

import nnunetv2.paths as paths
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def generate_low_data_splits():
    for dataset in [216,]:
        # add folds to splits_final.json of d64 that reflect annotated dataset percentage
        splits_final_file = join(paths.nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset), 'splits_final.json')
        splits = load_json(splits_final_file)
        num_cases = len(splits[0]['train']) + len(splits[0]['val'])
        splits = splits[:5]
        # assert len(splits) == 5

        # add 5 folds for 10%. Keep val set of fold 0
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 10), replace=False)),
                 'val': splits[0]['val']}
            )
        # add 5 folds with 3%
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 33), replace=False)),
                 'val': splits[0]['val']}
            )
        # add 5 folds with 5%
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases * 0.05), replace=False)),
                 'val': splits[0]['val']}
            )
        # add 5 folds with 30%
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases * 0.3), replace=False)),
                 'val': splits[0]['val']}
            )

        # add 5 folds with 50%
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases * 0.5), replace=False)),
                 'val': splits[0]['val']}
            )

        save_json(splits, splits_final_file)


if __name__ == '__main__':
    generate_low_data_splits()
