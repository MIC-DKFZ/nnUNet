import os
from typing import Optional

import requests
from batchgenerators.utilities.file_and_folder_operations import *
from tempfile import TemporaryDirectory
from nnunetv2.model_sharing.model_import import install_model_from_zip_file
from nnunetv2.paths import nnUNet_results
from tqdm import tqdm


def download_and_install_from_url(url):
    nnUNet_results.require()
    print('Downloading pretrained model from url:', url)
    home = os.path.expanduser('~')
    with TemporaryDirectory(prefix='.nnunetdownload_', dir=home) as tmpdir:
        zip_path = join(tmpdir, 'model.zip')
        download_file(url=url, local_filename=zip_path, chunk_size=8192 * 16)
        print("Download finished. Extracting...")
        install_model_from_zip_file(zip_path)
        print("Done")


def download_file(url: str, local_filename: str, chunk_size: Optional[int] = 8192 * 16) -> str:
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        # tqdm.wrapattr does not close the file. Close it ourselves, also when the download fails, otherwise the
        # caller cannot delete it on Windows
        with open(local_filename, 'wb') as fh, \
                tqdm.wrapattr(fh, "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename
