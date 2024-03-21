import shutil
from typing import Union, List, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, join, isdir, maybe_mkdir_p, subfiles, isfile

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
import nnunetv2.paths as paths
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def accumulate_cv_results(trained_model_folder,
                          merged_output_folder: str,
                          folds: Union[List[int], Tuple[int, ...]],
                          num_processes: int = default_num_processes,
                          overwrite: bool = True):
    """
    There are a lot of things that can get fucked up, so the simplest way to deal with potential problems is to
    collect the cv results into a separate folder and then evaluate them again. No messing with summary_json files!
    """

    if overwrite and isdir(merged_output_folder):
        shutil.rmtree(merged_output_folder)
    maybe_mkdir_p(merged_output_folder)

    dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    rw = plans_manager.image_reader_writer_class()
    shutil.copy(join(trained_model_folder, 'dataset.json'), join(merged_output_folder, 'dataset.json'))
    shutil.copy(join(trained_model_folder, 'plans.json'), join(merged_output_folder, 'plans.json'))

    did_we_copy_something = False
    for f in folds:
        expected_validation_folder = join(trained_model_folder, f'fold_{f}', 'validation')
        if not isdir(expected_validation_folder):
            raise RuntimeError(f"fold {f} of model {trained_model_folder} is missing. Please train it!")
        predicted_files = subfiles(expected_validation_folder, suffix=dataset_json['file_ending'], join=False)
        for pf in predicted_files:
            if overwrite and isfile(join(merged_output_folder, pf)):
                raise RuntimeError(f'More than one of your folds has a prediction for case {pf}')
            if overwrite or not isfile(join(merged_output_folder, pf)):
                shutil.copy(join(expected_validation_folder, pf), join(merged_output_folder, pf))
                did_we_copy_something = True

    if did_we_copy_something or not isfile(join(merged_output_folder, 'summary.json')):
        label_manager = plans_manager.get_label_manager(dataset_json)
        gt_folder = join(paths.nnUNet_raw, plans_manager.dataset_name, 'labelsTr')
        if not isdir(gt_folder):
            gt_folder = join(paths.nnUNet_preprocessed, plans_manager.dataset_name, 'gt_segmentations')
        compute_metrics_on_folder(gt_folder,
                                  merged_output_folder,
                                  join(merged_output_folder, 'summary.json'),
                                  rw,
                                  dataset_json['file_ending'],
                                  label_manager.foreground_regions if label_manager.has_regions else
                                  label_manager.foreground_labels,
                                  label_manager.ignore_label,
                                  num_processes)
