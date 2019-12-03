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

from multiprocessing.pool import Pool
import shutil
import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
import argparse
from nnunet.postprocessing.connected_components import determine_postprocessing


def merge(args):
    file1, file2, properties_file, out_file = args
    if not isfile(out_file):
        res1 = np.load(file1)['softmax']
        res2 = np.load(file2)['softmax']
        props = load_pickle(properties_file)
        mn = np.mean((res1, res2), 0)
        save_segmentation_nifti_from_softmax(mn, out_file, props, 3, None, None, None, force_separate_z=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="This is intended to ensemble training images (from cross-validation) only. Use"
                                           "inference/ensemble_predictions.py instead")
    parser.add_argument("training_output_folder1")
    parser.add_argument("training_output_folder2")
    parser.add_argument("output_folder")
    parser.add_argument("task") # we need to know this for gt_segmentations
    parser.add_argument("validation_folder")

    args = parser.parse_args()

    training_output_folder1 = args.training_output_folder1
    training_output_folder2 = args.training_output_folder2
    output_folder_base = args.output_folder
    output_folder = join(output_folder_base, "ensembled_raw")
    task = args.task
    validation_folder = args.validation_folder

    # only_keep_largest_connected_component is the same for all stages
    dataset_directory = join(preprocessing_output_dir, task)
    plans = load_pickle(join(training_output_folder1, "plans.pkl"))  # we need this only for the labels

    files1 = []
    files2 = []
    property_files = []
    out_files = []
    gt_segmentations = []

    folder_with_gt_segs = join(dataset_directory, "gt_segmentations")
    # in the correct shape and we need the original geometry to restore the niftis

    folds = np.arange(5)

    for f in folds:
        validation_folder_net1 = join(training_output_folder1, "fold_%d" % f, validation_folder)
        validation_folder_net2 = join(training_output_folder2, "fold_%d" % f, validation_folder)
        patient_identifiers1 = subfiles(validation_folder_net1, False, None, 'npz', True)
        patient_identifiers2 = subfiles(validation_folder_net2, False, None, 'npz', True)
        # we don't do postprocessing anymore so there should not be any of that noPostProcess
        patient_identifiers1_nii = [i for i in subfiles(validation_folder_net1, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]
        patient_identifiers2_nii = [i for i in subfiles(validation_folder_net2, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]
        assert len(patient_identifiers1) == len(patient_identifiers1_nii), "npz seem to be missing. run validation with --npz"
        assert len(patient_identifiers1) == len(patient_identifiers1_nii), "npz seem to be missing. run validation with --npz"
        assert all([i[:-4] == j[:-7] for i, j in zip(patient_identifiers1, patient_identifiers1_nii)]), "npz seem to be missing. run validation with --npz"
        assert all([i[:-4] == j[:-7] for i, j in zip(patient_identifiers2, patient_identifiers2_nii)]), "npz seem to be missing. run validation with --npz"

        all_patient_identifiers = patient_identifiers1
        for p in patient_identifiers2:
            if p not in all_patient_identifiers:
                all_patient_identifiers.append(p)

        # assert these patients exist for both methods
        assert all([isfile(join(validation_folder_net1, i)) for i in all_patient_identifiers])
        assert all([isfile(join(validation_folder_net2, i)) for i in all_patient_identifiers])

        maybe_mkdir_p(output_folder)

        for p in all_patient_identifiers:
            files1.append(join(validation_folder_net1, p))
            files2.append(join(validation_folder_net2, p))
            property_files.append(join(validation_folder_net1, p)[:-3] + "pkl")
            out_files.append(join(output_folder, p[:-4] + ".nii.gz"))
            gt_segmentations.append(join(folder_with_gt_segs, p[:-4] + ".nii.gz"))

    p = Pool(default_num_threads)
    p.map(merge, zip(files1, files2, property_files, out_files))
    p.close()
    p.join()

    if not isfile(join(output_folder, "summary.json")) and len(out_files) > 0:
        aggregate_scores(tuple(zip(out_files, gt_segmentations)), labels=plans['all_classes'],
                     json_output_file=join(output_folder, "summary.json"), json_task=task,
                     json_name=task + "__" + output_folder_base.split("/")[-1], num_threads=default_num_threads)

    if not isfile(join(output_folder_base, "postprocessing.json")):
        # now lets also look at postprocessing. We cannot just take what we determined in cross-validation and apply it
        # here because things may have changed and may also be too inconsistent between the two networks
        determine_postprocessing(output_folder_base, folder_with_gt_segs, "ensembled_raw", "temp",
                                 "ensembled_postprocessed", default_num_threads, dice_threshold=0)

        out_dir_all_json = join(network_training_output_dir, "summary_jsons")
        json_out = load_json(join(output_folder_base, "ensembled_postprocessed", "summary.json"))

        json_out["experiment_name"] = output_folder_base.split("/")[-1]
        save_json(json_out, join(output_folder_base, "ensembled_postprocessed", "summary.json"))

        shutil.copy(join(output_folder_base, "ensembled_postprocessed", "summary.json"),
                    join(out_dir_all_json, "%s__%s.json" % (task, output_folder_base.split("/")[-1])))
