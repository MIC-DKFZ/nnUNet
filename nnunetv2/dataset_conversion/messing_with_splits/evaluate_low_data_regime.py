import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

def generate_low_data_splits():
    for dataset in [216, 994]:
        # add folds to splits_final.json of d64 that reflect annotated dataset percentage
        splits_final_file = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset), 'splits_final.json')
        splits = load_json(splits_final_file)
        num_cases = len(splits[0]['train']) + len(splits[0]['val'])
        # add 5 folds for 10%. Keep val set of fold 0
        assert len(splits) == 5
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 10))),
                 'val': splits[0]['val']}
            )
        # add 5 folds with 3%
        for n in range(5):
            splits.append(
                {'train': list(np.random.choice(splits[0]['train'], size=round(num_cases / 33))),
                 'val': splits[0]['val']}
            )
        save_json(splits, splits_final_file)


def collect_results():
    configurations_all = {
        216: ("3d_lowres", "3d_lowres_sparse_slicewise_10", "3d_lowres_sparse_slicewise_30", "3d_lowres_sparse_blobs", "3d_lowres_sparse_randblobs", "3d_lowres_sparse_pixelwise"),
        994: ("3d_fullres", "3d_fullres_sparse_slicewise_10", "3d_fullres_sparse_slicewise_30", "3d_fullres_sparse_blobs", "3d_fullres_sparse_randblobs", "3d_fullres_sparse_pixelwise"),
    }

    pass


if __name__ == '__main__':
    pass
