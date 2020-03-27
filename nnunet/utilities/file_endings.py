from batchgenerators.utilities.file_and_folder_operations import *


def remove_trailing_slash(filename: str):
    while filename.endswith('/'):
        filename = filename[:-1]
    return filename


def maybe_add_0000_to_all_niigz(folder):
    nii_gz = subfiles(folder, suffix='.nii.gz')
    for n in nii_gz:
        n = remove_trailing_slash(n)
        os.rename(n, n[:-7] + '_0000.nii.gz')
