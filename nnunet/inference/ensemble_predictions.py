#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool


def merge_files(args):
    files, properties_file, out_file, only_keep_largest_connected_component, min_region_size_per_class, override = args
    if override or not isfile(out_file):
        softmax = [np.load(f)['softmax'][None] for f in files]
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0)
        props = load_pickle(properties_file)
        save_segmentation_nifti_from_softmax(softmax, out_file, props, 1, None, None, None)


def merge(folders, output_folder, threads, override=True):
    maybe_mkdir_p(output_folder)

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        assert all([isfile(join(f, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                        "all folders"
        assert all([isfile(join(f, i + ".pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append([join(f, p + ".npz") for f in folders])
        property_files.append(join(folders[0], p + ".pkl"))
        out_files.append(join(output_folder, p + ".nii.gz"))

    plans = load_pickle(join(folders[0], "plans.pkl"))

    only_keep_largest_connected_component, min_region_size_per_class = plans['keep_only_largest_region'], \
                                                                       plans['min_region_size_per_class']
    p = Pool(threads)
    p.map(merge_files, zip(files, property_files, out_files, [only_keep_largest_connected_component] * len(out_files),
                           [min_region_size_per_class] * len(out_files), [override] * len(out_files)))
    p.close()
    p.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This requires that all folders to be merged use the same "
                                                 "postprocessing function "
                                                 "(nnunet.utilities.postprocessing.postprocess_segmentation). "
                                                 "This will be the case if the corresponding "
                                                 "models were trained with nnUNetTrainer or nnUNetTrainerCascadeFullRes"
                                                 "but may not be the case if you added models of your own that use a "
                                                 "different postprocessing. This script also requires a plans file to"
                                                 "be present in all of the folders (if they are not present you can "
                                                 "take them from the respective model training output folders. "
                                                 "Parameters for the postprocessing "
                                                 "will be taken from the plans file. If the folders were created by "
                                                 "predict_folder.py then the plans file will have been copied "
                                                 "automatically (if --save_npz is specified)")
    parser.add_argument('-f', '--folders', nargs='+', help="list of folders to merge. All folders must contain npz "
                                                           "files", required=True)
    parser.add_argument('-o', '--output_folder', help="where to save the results", required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=2,
                        type=int)

    args = parser.parse_args()

    folders = args.folders
    threads = args.threads
    output_folder = args.output_folder

    merge(folders, output_folder, threads, override=True)
