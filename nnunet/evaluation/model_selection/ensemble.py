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
from multiprocessing.pool import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.paths import network_training_output_dir, preprocessing_output_dir
from nnunet.postprocessing.connected_components import determine_postprocessing


def merge(args):
    file1, file2, properties_file, out_file = args
    if not isfile(out_file):
        res1 = np.load(file1)['softmax']
        res2 = np.load(file2)['softmax']
        props = load_pickle(properties_file)
        mn = np.mean((res1, res2), 0)
        # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
        # don't matter here)
        save_segmentation_nifti_from_softmax(mn, out_file, props, 3, None, None, None, force_separate_z=None,
                                             interpolation_order_z=0)


def ensemble(training_output_folder1, training_output_folder2, output_folder, task, validation_folder, folds, allow_ensembling: bool = True):
    print("\nEnsembling folders\n", training_output_folder1, "\n", training_output_folder2)

    output_folder_base = output_folder
    output_folder = join(output_folder_base, "ensembled_raw")

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

    for f in folds:
        validation_folder_net1 = join(training_output_folder1, "fold_%d" % f, validation_folder)
        validation_folder_net2 = join(training_output_folder2, "fold_%d" % f, validation_folder)

        if not isdir(validation_folder_net1):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `nnUNet_train CONFIG TRAINER TASK FOLD -val --npz`" % validation_folder_net1)
        if not isdir(validation_folder_net2):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `nnUNet_train CONFIG TRAINER TASK FOLD -val --npz`" % validation_folder_net2)

        # we need to ensure the validation was successful. We can verify this via the presence of the summary.json file
        if not isfile(join(validation_folder_net1, 'summary.json')):
            raise AssertionError("Validation directory incomplete: %s. Please rerun validation with `nnUNet_train CONFIG TRAINER TASK FOLD -val --npz`" % validation_folder_net1)
        if not isfile(join(validation_folder_net2, 'summary.json')):
            raise AssertionError("Validation directory missing: %s. Please rerun validation with `nnUNet_train CONFIG TRAINER TASK FOLD -val --npz`" % validation_folder_net2)

        patient_identifiers1_npz = [i[:-4] for i in subfiles(validation_folder_net1, False, None, 'npz', True)]
        patient_identifiers2_npz = [i[:-4] for i in subfiles(validation_folder_net2, False, None, 'npz', True)]

        # we don't do postprocessing anymore so there should not be any of that noPostProcess
        patient_identifiers1_nii = [i[:-7] for i in subfiles(validation_folder_net1, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]
        patient_identifiers2_nii = [i[:-7] for i in subfiles(validation_folder_net2, False, None, suffix='nii.gz', sort=True) if not i.endswith("noPostProcess.nii.gz") and not i.endswith('_postprocessed.nii.gz')]

        if not all([i in patient_identifiers1_npz for i in patient_identifiers1_nii]):
            raise AssertionError("Missing npz files in folder %s. Please run the validation for all models and folds with the '--npz' flag." % (validation_folder_net1))
        if not all([i in patient_identifiers2_npz for i in patient_identifiers2_nii]):
            raise AssertionError("Missing npz files in folder %s. Please run the validation for all models and folds with the '--npz' flag." % (validation_folder_net2))

        patient_identifiers1_npz.sort()
        patient_identifiers2_npz.sort()

        assert all([i == j for i, j in zip(patient_identifiers1_npz, patient_identifiers2_npz)]), "npz filenames do not match. This should not happen."

        maybe_mkdir_p(output_folder)

        for p in patient_identifiers1_npz:
            files1.append(join(validation_folder_net1, p + '.npz'))
            files2.append(join(validation_folder_net2, p + '.npz'))
            property_files.append(join(validation_folder_net1, p) + ".pkl")
            out_files.append(join(output_folder, p + ".nii.gz"))
            gt_segmentations.append(join(folder_with_gt_segs, p + ".nii.gz"))

    p = Pool(default_num_threads)
    p.map(merge, zip(files1, files2, property_files, out_files))
    p.close()
    p.join()

    if not isfile(join(output_folder, "summary.json")) and len(out_files) > 0:
        aggregate_scores(tuple(zip(out_files, gt_segmentations)), labels=plans['all_classes'],
                     json_output_file=join(output_folder, "summary.json"), json_task=task,
                     json_name=task + "__" + os.path.basename(output_folder_base), num_threads=default_num_threads)

    if allow_ensembling and not isfile(join(output_folder_base, "postprocessing.json")):
        # now lets also look at postprocessing. We cannot just take what we determined in cross-validation and apply it
        # here because things may have changed and may also be too inconsistent between the two networks
        determine_postprocessing(output_folder_base, folder_with_gt_segs, "ensembled_raw", "temp",
                                 "ensembled_postprocessed", default_num_threads, dice_threshold=0)

        out_dir_all_json = join(network_training_output_dir, "summary_jsons")
        json_out = load_json(join(output_folder_base, "ensembled_postprocessed", "summary.json"))

        json_out["experiment_name"] = os.path.basename(output_folder_base)
        save_json(json_out, join(output_folder_base, "ensembled_postprocessed", "summary.json"))

        maybe_mkdir_p(out_dir_all_json)
        shutil.copy(join(output_folder_base, "ensembled_postprocessed", "summary.json"),
                    join(out_dir_all_json, "%s__%s.json" % (task, os.path.basename(output_folder_base))))
