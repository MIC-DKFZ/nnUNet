from typing import Optional

import requests
from batchgenerators.utilities.file_and_folder_operations import *
from time import time
from nnunetv2.model_sharing.model_import import install_model_from_zip_file
from nnunetv2.paths import nnUNet_results
from tqdm import tqdm


def download_and_install_from_url(url):
    assert nnUNet_results is not None, "Cannot install model because network_training_output_dir is not " \
                                                    "set (RESULTS_FOLDER missing as environment variable, see " \
                                                    "Installation instructions)"
    print('Downloading pretrained model from url:', url)
    import http.client
    http.client.HTTPConnection._http_vsn = 10
    http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    import os
    home = os.path.expanduser('~')
    random_number = int(time() * 1e7)
    tempfile = join(home, '.nnunetdownload_%s' % str(random_number))

    try:
        download_file(url=url, local_filename=tempfile, chunk_size=8192 * 16)
        print("Download finished. Extracting...")
        install_model_from_zip_file(tempfile)
        print("Done")
    except Exception as e:
        raise e
    finally:
        if isfile(tempfile):
            os.remove(tempfile)


def download_file(url: str, local_filename: str, chunk_size: Optional[int] = 8192 * 16) -> str:
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(local_filename, 'wb'), "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename


