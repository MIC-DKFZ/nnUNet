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
from typing import Tuple

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.postprocessing.connected_components import determine_postprocessing
import argparse


def collect_cv_niftis(cv_folder: str, output_folder: str, validation_folder_name: str = 'validation_raw',
                      folds: tuple = (0, 1, 2, 3, 4)):
    validation_raw_folders = [join(cv_folder, "fold_%d" % i, validation_folder_name) for i in folds]
    exist = [isdir(i) for i in validation_raw_folders]

    if not all(exist):
        raise RuntimeError("some folds are missing. Please run the full 5-fold cross-validation. "
                           "The following folds seem to be missing: %s" %
                           [i for j, i in enumerate(folds) if not exist[j]])

    # now copy all raw niftis into cv_niftis_raw
    maybe_mkdir_p(output_folder)
    for f in folds:
        niftis = subfiles(validation_raw_folders[f], suffix=".nii.gz")
        for n in niftis:
            shutil.copy(n, join(output_folder))


def consolidate_folds(output_folder_base, validation_folder_name: str = 'validation_raw',
                      advanced_postprocessing: bool = False, folds: Tuple[int] = (0, 1, 2, 3, 4)):
    """
    Used to determine the postprocessing for an experiment after all five folds have been completed. In the validation of
    each fold, the postprocessing can only be determined on the cases within that fold. This can result in different
    postprocessing decisions for different folds. In the end, we can only decide for one postprocessing per experiment,
    so we have to rerun it
    :param folds:
    :param advanced_postprocessing:
    :param output_folder_base:experiment output folder (fold_0, fold_1, etc must be subfolders of the given folder)
    :param validation_folder_name: dont use this
    :return:
    """
    output_folder_raw = join(output_folder_base, "cv_niftis_raw")
    if isdir(output_folder_raw):
        shutil.rmtree(output_folder_raw)

    output_folder_gt = join(output_folder_base, "gt_niftis")
    collect_cv_niftis(output_folder_base, output_folder_raw, validation_folder_name,
                      folds)

    num_niftis_gt = len(subfiles(join(output_folder_base, "gt_niftis"), suffix='.nii.gz'))
    # count niftis in there
    num_niftis = len(subfiles(output_folder_raw, suffix='.nii.gz'))
    if num_niftis != num_niftis_gt:
        raise AssertionError("If does not seem like you trained all the folds! Train all folds first!")

    # load a summary file so that we can know what class labels to expect
    summary_fold0 = load_json(join(output_folder_base, "fold_0", validation_folder_name, "summary.json"))['results'][
        'mean']
    classes = [int(i) for i in summary_fold0.keys()]
    niftis = subfiles(output_folder_raw, join=False, suffix=".nii.gz")
    test_pred_pairs = [(join(output_folder_gt, i), join(output_folder_raw, i)) for i in niftis]

    # determine_postprocessing needs a summary.json file in the folder where the raw predictions are. We could compute
    # that from the summary files of the five folds but I am feeling lazy today
    aggregate_scores(test_pred_pairs, labels=classes, json_output_file=join(output_folder_raw, "summary.json"),
                     num_threads=default_num_threads)

    determine_postprocessing(output_folder_base, output_folder_gt, 'cv_niftis_raw',
                             final_subf_name="cv_niftis_postprocessed", processes=default_num_threads,
                             advanced_postprocessing=advanced_postprocessing)
    # determine_postprocessing will create a postprocessing.json file that can be used for inference


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", type=str, required=True, help="experiment output folder (fold_0, fold_1, "
                                                               "etc must be subfolders of the given folder)")

    args = argparser.parse_args()

    folder = args.f

    consolidate_folds(folder)
