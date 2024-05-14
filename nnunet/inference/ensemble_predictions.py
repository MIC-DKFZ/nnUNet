#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import shutil
from copy import deepcopy

from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing


def merge_files(files, properties_files, out_file, override, store_npz):
    if not override and isfile(out_file):
        return
    softmax = [np.load(f)['softmax'][None] for f in files]
    softmax = np.vstack(softmax)
    softmax = np.mean(softmax, 0)
    props = [load_pickle(f) for f in properties_files]

    reg_class_orders = [p['regions_class_order'] if 'regions_class_order' in p.keys() else None
                        for p in props]

    if any(i is not None for i in reg_class_orders):
        # if reg_class_orders are not None then they must be the same in all pkls
        tmp = reg_class_orders[0]
        for r in reg_class_orders[1:]:
            assert tmp == r, 'If merging files with regions_class_order, the regions_class_orders of all ' \
                             'files must be the same. regions_class_order: %s, \n files: %s' % \
                             (str(reg_class_orders), str(files))
        regions_class_order = tmp
    else:
        regions_class_order = None

    # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
    # don't matter here)
    save_segmentation_nifti_from_softmax(softmax, out_file, props[0], 3, regions_class_order, None, None,
                                         force_separate_z=None)
    if store_npz:
        np.savez_compressed(out_file[:-7] + ".npz", softmax=softmax)
        save_pickle(props, out_file[:-7] + ".pkl")


def merge(folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):
    # maybe_mkdir_p(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if postprocessing_file is not None:
        output_folder_orig = deepcopy(output_folder)
        output_folder = join(output_folder, 'not_postprocessed')
        # maybe_mkdir_p(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder_orig = None

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        assert all(isfile(join(f, i + ".npz")) for i in patient_ids), (
            "Not all patient npz are available in " "all folders"
        )
        assert all(isfile(join(f, i + ".pkl")) for i in patient_ids), (
            "Not all patient pkl are available in " "all folders"
        )

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append([join(f, p + ".npz") for f in folders])
        property_files.append([join(f, p + ".pkl") for f in folders])
        out_files.append(join(output_folder, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(merge_files, zip(files, property_files, out_files, [override] * len(out_files), [store_npz] * len(out_files)))
    p.close()
    p.join()

    if postprocessing_file is not None:
        for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
        print('Postprocessing...')
        apply_postprocessing_to_folder(output_folder, output_folder_orig,
                                       for_which_classes, min_valid_obj_size, threads)
        shutil.copy(postprocessing_file, output_folder_orig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="This script will merge predictions (that were prdicted with the "
                                                 "-npz option!). You need to specify a postprocessing file so that "
                                                 "we know here what postprocessing must be applied. Failing to do so "
                                                 "will disable postprocessing")
    parser.add_argument('-f', '--folders', nargs='+', help="list of folders to merge. All folders must contain npz "
                                                           "files", required=True)
    parser.add_argument('-o', '--output_folder', help="where to save the results", required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=2,
                        type=int)
    parser.add_argument('-pp', '--postprocessing_file', help="path to the file where the postprocessing configuration "
                                                             "is stored. If this is not provided then no postprocessing "
                                                             "will be made. It is strongly recommended to provide the "
                                                             "postprocessing file!",
                        required=False, type=str, default=None)
    parser.add_argument('--npz', action="store_true", required=False, help="stores npz and pkl")

    args = parser.parse_args()

    folders = args.folders
    threads = args.threads
    output_folder = args.output_folder
    pp_file = args.postprocessing_file
    npz = args.npz

    merge(folders, output_folder, threads, override=True, postprocessing_file=pp_file, store_npz=npz)


if __name__ == "__main__":
    main()
