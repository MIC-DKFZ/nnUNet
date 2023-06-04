import argparse
import yaml
import pickle

import numpy as np
import multiprocessing as mp
import SimpleITK as sitk

from functools import partial
from math import ceil
from pathlib import Path
from tqdm import tqdm
from typing import Union

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import load_json, join
import nnunetv2.training.dataloading.utils as nnutils


def processing(file_path: Path, out_path: Path, cfg: dict):

    extension = ('').join(file_path.suffixes)
    filename = file_path.name.replace(extension, '')
    
    # read image
    if extension == '.nii.gz':
        imageITK = sitk.ReadImage(str(file_path))
        image = sitk.GetArrayFromImage(imageITK)
        # get metadata
        ori_spacing = np.asarray(imageITK.GetSpacing())[[2, 1, 0]]
        ori_origin = imageITK.GetOrigin()
        ori_direction = imageITK.GetDirection()
    elif extension == '.npy':
        image = np.load(file_path)
        with open(file_path.parent/f'{file_path.name.rstrip(".npy")}.pkl', 'rb') as pfile:
            metadata = pickle.load(pfile)
            ori_spacing = metadata['sitk_stuff']['spacing']
            ori_origin = metadata['sitk_stuff']['origin']
            ori_direction = metadata['sitk_stuff']['direction']

    # get crop dimensions
    _, d, w, h = image.shape
    tile_size = cfg['tile_size']
    stride_d = cfg['stride_d']
    tile_deps = int(ceil((d - tile_size) / stride_d) + 1)

    # this just tiles the images across the z/depth direction
    for dep in range(tile_deps):
        path_dep = out_path / f'{filename}_dep{dep}.nii.gz'
        if path_dep.exists():
            continue
        else:
            d1 = int(dep * stride_d)
            d2 = min(d1 + tile_size, d)
            if d2-d1 < tile_size:
                d1 = d2-tile_size

            # If the image hight and width are too big, crop them to the 80%
            if w > 320:
                img = image[0, 
                    np.maximum(d1, 0):d2, int(w * 0.1):int(w * 0.9), int(h * 0.1):int(h * 0.9)]
            else:
                img = image[0, np.maximum(d1, 0):d2, :, :]

            img = img.astype('float32')

            # save image
            path_dep.parent.mkdir(exist_ok=True, parents=True)
            saveITK = sitk.GetImageFromArray(img)
            saveITK.SetSpacing(np.asarray(ori_spacing)[[2, 1, 0]])
            saveITK.SetOrigin(ori_origin)
            saveITK.SetDirection(ori_direction)
            sitk.WriteImage(saveITK, str(path_dep))


def main(
    dataset_name_or_id: Union[str, int],
    plans_identifier: str,
    nnu_configuration: str,
    cfg_file: str,
    n_processes: int = mp.cpu_count()
):
    # get ssl configurations
    with open(cfg_file, 'r') as yfile:
        ssl_cfg = yaml.safe_load(yfile)

    # get nnUnet dataset files/configurations
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    plans_file = Path(nnUNet_preprocessed)/dataset_name/f'{plans_identifier}.json'
    # plans = load_json(str(plans_file))
    # pp_data_folder = plans['configurations'][nnu_configuration]['data_identifier'][]
    pp_data_folder = f'{plans_identifier}_{nnu_configuration}'

    # get nnUnet preprocessed files/configurations
    preprocessed_path = Path(nnUNet_preprocessed)/dataset_name/pp_data_folder
    dataset_json = load_json(str(preprocessed_path.parent/'dataset.json'))
    n_train_imgs = dataset_json['numTraining']

    # get nnUnet preprocessed files names
    pp_files = [file for file in preprocessed_path.iterdir() if (file.suffix == '.npy') and ('seg' not in file.name)]
    res = []
    if len(pp_files) == 0:
        pp_files = [str(file) for file in preprocessed_path.iterdir() if (file.suffix == '.npz') and ('seg' not in file.name)]
        with mp.Pool(mp.cpu_count()) as pool:
            for result in pool.imap(nnutils._convert_to_npy, pp_files):
                res.append(result)
    pp_files = [file for file in preprocessed_path.iterdir() if (file.suffix == '.npy') and ('seg' not in file.name)]
    assert len(pp_files) == n_train_imgs, \
        f'number of files in preprocessed dir, does not match plan: {len(pp_files)} and {n_train_imgs}'
    
    # get crops path
    output_path = Path(nnUNet_preprocessed)/dataset_name/f'{pp_data_folder}_ssl'

    # get the crops
    res = []
    with mp.Pool(mp.cpu_count()) as pool:
        prep_fn = partial(processing, out_path=output_path, cfg=ssl_cfg['cropping'])
        for result in tqdm(pool.imap(prep_fn, pp_files), total=len(pp_files)):
            res.append(result)

    # save the filenames
    file_path = preprocessed_path.parent/'ssl_dataset.txt'
    file = open(file_path, "w")
    for name in sorted(list(output_path.iterdir())):
        if str(name).endswith(".nii.gz"):
            file.write(f'{str(name)}\n')
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset_name', help='Dataset name')
    parser.add_argument('-p', dest='planner', help='nnUnetPlanner name')
    parser.add_argument('-c', dest='nnu_configuration', help='nnUnet configuration')
    parser.add_argument('-np', dest='n_processes', help='number of processors to use')
    parser.add_argument('-cfg', dest='config_path', help='path to configuration file')
    args = parser.parse_args()
    main(
        args.dataset_name,
        args.planner,
        args.nnu_configuration,
        args.config_path,
        args.n_processes
    )
