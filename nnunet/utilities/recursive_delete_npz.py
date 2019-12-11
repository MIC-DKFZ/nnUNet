from batchgenerators.utilities.file_and_folder_operations import *
import argparse
import os


def recursive_delete_npz(current_directory: str):
    npz_files = subfiles(current_directory, join=True, suffix=".npz")
    npz_files = [i for i in npz_files if not i.endswith("segFromPrevStage.npz")] # to be extra safe
    _ = [os.remove(i) for i in npz_files]
    for d in subdirs(current_directory, join=False):
        if d != "pred_next_stage":
            recursive_delete_npz(join(current_directory, d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="USE THIS RESPONSIBLY! DANGEROUS! I (Fabian) use this to remove npz files "
                                           "after I ran figure_out_what_to_submit")
    parser.add_argument("-f", help="folder", required=True)

    args = parser.parse_args()

    recursive_delete_npz(args.f)
