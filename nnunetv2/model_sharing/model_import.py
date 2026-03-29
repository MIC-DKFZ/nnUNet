import zipfile

from nnunetv2.paths import require_results_path


def install_model_from_zip_file(zip_file: str):
    results_dir = require_results_path('installing pretrained models')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(results_dir)
