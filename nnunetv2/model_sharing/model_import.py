import zipfile

import nnunetv2.paths as paths


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(paths.nnUNet_results)
