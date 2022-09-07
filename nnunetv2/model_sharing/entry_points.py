from nnunetv2.model_sharing.model_download import download_and_install_from_url
from nnunetv2.model_sharing.model_import import install_model_from_zip_file


def print_license_warning():
    print('')
    print('######################################################')
    print('!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!')
    print('######################################################')
    print("Using the pretrained model weights is subject to the license of the dataset they were trained on. Some "
          "allow commercial use, others don't. It is your responsibility to make sure you use them appropriately! Use "
          "nnUNet_print_pretrained_model_info(task_name) to see a summary of the dataset and where to find its license!")
    print('######################################################')
    print('')


def download_by_url():
    import argparse
    parser = argparse.ArgumentParser(
        description="Use this to download pretrained models. This script is intended to download models via url only. "
                    "CAREFUL: This script will overwrite "
                    "existing models (if they share the same trainer class and plans as "
                    "the pretrained model.")
    parser.add_argument("url", type=str, help='URL of the pretrained model')
    args = parser.parse_args()
    url = args.url
    download_and_install_from_url(url)


def install_from_zip_entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description="Use this to install a zip file containing a pretrained model.")
    parser.add_argument("zip", type=str, help='zip file')
    args = parser.parse_args()
    zip = args.zip
    install_model_from_zip_file(zip)


