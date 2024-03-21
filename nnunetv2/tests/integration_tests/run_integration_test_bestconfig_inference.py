import argparse

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

from nnunetv2.ensembling.ensemble import ensemble_folders
from nnunetv2.evaluation.find_best_configuration import find_best_configuration, \
    dumb_trainer_config_plans_to_trained_models_dict
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nnunetv2.paths as paths
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder


if __name__ == '__main__':
    """
    Predicts the imagesTs folder with the best configuration and applies postprocessing
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, help='dataset id')
    args = parser.parse_args()
    d = args.d

    dataset_name = maybe_convert_to_dataset_name(d)
    source_dir = join(paths.nnUNet_raw, dataset_name, 'imagesTs')
    target_dir_base = join(paths.nnUNet_results, dataset_name)

    models = dumb_trainer_config_plans_to_trained_models_dict(['nnUNetTrainer_5epochs'],
                                                              ['2d',
                                                               '3d_lowres',
                                                               '3d_cascade_fullres',
                                                               '3d_fullres'],
                                                              ['nnUNetPlans'])
    ret = find_best_configuration(d, models, allow_ensembling=True, num_processes=8, overwrite=True,
                                  folds=(0, 1, 2, 3, 4), strict=True)

    has_ensemble = len(ret['best_model_or_ensemble']['selected_model_or_models']) > 1

    # we don't use all folds to speed stuff up
    used_folds = (0, 3)
    output_folders = []
    for im in ret['best_model_or_ensemble']['selected_model_or_models']:
        output_dir = join(target_dir_base, f"pred_{im['configuration']}")
        model_folder = get_output_folder(d, im['trainer'], im['plans_identifier'], im['configuration'])
        # note that if the best model is the enseble of 3d_lowres and 3d cascade then 3d_lowres will be predicted
        # twice (once standalone and once to generate the predictions for the cascade) because we don't reuse the
        # prediction here. Proper way would be to check for that and
        # then give the output of 3d_lowres inference to the folder_with_segs_from_prev_stage kwarg in
        # predict_from_raw_data. Since we allow for
        # dynamically setting 'previous_stage' in the plans I am too lazy to implement this here. This is just an
        # integration test after all. Take a closer look at how this in handled in predict_from_raw_data
        predictor = nnUNetPredictor(verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(model_folder, used_folds)
        predictor.predict_from_files(source_dir, output_dir, has_ensemble, overwrite=True)
        # predict_from_raw_data(list_of_lists_or_source_folder=source_dir, output_folder=output_dir,
        #                       model_training_output_dir=model_folder, use_folds=used_folds,
        #                       save_probabilities=has_ensemble, verbose=False, overwrite=True)
        output_folders.append(output_dir)

    # if we have an ensemble, we need to ensemble the results
    if has_ensemble:
        ensemble_folders(output_folders, join(target_dir_base, 'ensemble_predictions'), save_merged_probabilities=False)
        folder_for_pp = join(target_dir_base, 'ensemble_predictions')
    else:
        folder_for_pp = output_folders[0]

    # apply postprocessing
    pp_fns, pp_fn_kwargs = load_pickle(ret['best_model_or_ensemble']['postprocessing_file'])
    apply_postprocessing_to_folder(folder_for_pp, join(target_dir_base, 'ensemble_predictions_postprocessed'),
                                   pp_fns,
                                   pp_fn_kwargs, plans_file_or_dict=ret['best_model_or_ensemble']['some_plans_file'])
