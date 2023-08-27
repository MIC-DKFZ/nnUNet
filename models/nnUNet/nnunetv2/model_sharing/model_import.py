import zipfile

from nnunetv2.paths import nnUNet_results


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(nnUNet_results)