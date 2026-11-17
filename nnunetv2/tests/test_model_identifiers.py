import os
import unittest
import zipfile
from tempfile import TemporaryDirectory
from unittest import mock

import nnunetv2.evaluation.find_best_configuration as find_best_configuration_module
import nnunetv2.model_sharing.model_export as model_export_module
import nnunetv2.utilities.file_path_utilities as file_path_utilities
from nnunetv2.utilities.file_path_utilities import convert_identifier_to_trainer_plans_config, \
    convert_trainer_plans_config_to_identifier, get_ensemble_name, parse_dataset_trainer_plans_configuration_from_path

# plans identifiers as nnUNetv2_preprocess_like_nnssl and nnUNetv2_plan_like_dynamic write them. They contain '__'
# (and '___'), which used to break every place that took a model or ensemble folder name apart (GitHub issue #3089)
LIKE_NNSSL_PLANS = 'ptPlans__nnFoundationCNN_like_pretrained____Spacing__1.00_1.00_1.00___Norm__Z'
DYNAMIC_PLANS = 'ptPlans_dynamic__nnFoundationCNN'
MODELS = [
    ('nnUNetTrainer', 'nnUNetPlans', '3d_fullres'),
    ('nnUNetTrainer_5epochs', 'nnUNetResEncUNetLPlans', '3d_cascade_fullres'),
    ('PretrainedTrainer', LIKE_NNSSL_PLANS, '3d_fullres'),
    ('DynamicPretrainedTrainer', DYNAMIC_PLANS, '3d_fullres'),
]


class TestModelIdentifierParsing(unittest.TestCase):
    def test_identifier_round_trip(self):
        for model in MODELS:
            identifier = convert_trainer_plans_config_to_identifier(*model)
            self.assertEqual(tuple(convert_identifier_to_trainer_plans_config(identifier)), model)
            self.assertEqual(tuple(convert_identifier_to_trainer_plans_config(f'/results/Dataset004_X/{identifier}')),
                             model)

    def test_parse_path(self):
        for model in MODELS:
            identifier = convert_trainer_plans_config_to_identifier(*model)
            for path in (f'/results/Dataset004_X/{identifier}', f'Dataset004_X/{identifier}/fold_all'):
                self.assertEqual(parse_dataset_trainer_plans_configuration_from_path(path), ('Dataset004_X', *model))

    def test_parse_path_without_model_folder_asserts(self):
        with self.assertRaises(AssertionError):
            parse_dataset_trainer_plans_configuration_from_path('/results/Dataset004_X/')


class TestFindBestConfigurationSelectedModels(unittest.TestCase):
    """find_best_configuration must report the selected models even when their plans identifiers contain '__'."""

    def _run(self, models, dice_single, dice_ensemble):
        def fake_summary(path):
            return {'foreground_mean': {'Dice': dice_ensemble if 'ensemble___' in path else dice_single}}

        with mock.patch.multiple(
                find_best_configuration_module,
                nnUNet_results='/results', nnUNet_preprocessed='/preprocessed',
                get_output_folder=lambda d, tr, pl, c, fold=None: f'/results/{d}/{tr}__{pl}__{c}',
                filter_available_models=lambda m, d: list(m),
                load_summary_json=fake_summary,
                load_json=lambda f: {'file_ending': '.nii.gz', 'foreground_mean': {'Dice': 1.}},
                accumulate_cv_results=mock.DEFAULT, ensemble_crossvalidations=mock.DEFAULT,
                PlansManager=mock.DEFAULT, compute_metrics_on_folder=mock.DEFAULT,
                determine_postprocessing=mock.DEFAULT, save_json=mock.DEFAULT,
                print_inference_instructions=mock.DEFAULT):
            res = find_best_configuration_module.find_best_configuration(
                'Dataset004_X', [{'trainer': tr, 'plans': pl, 'configuration': c} for tr, pl, c in models],
                allow_ensembling=len(models) > 1, folds=(0,))
        return [(m['trainer'], m['plans_identifier'], m['configuration'])
                for m in res['best_model_or_ensemble']['selected_model_or_models']]

    def test_single_model(self):
        model = ('PretrainedTrainer', LIKE_NNSSL_PLANS, '3d_fullres')
        self.assertEqual(self._run([model], 0.9, 0.), [model])

    def test_ensemble(self):
        models = [('PretrainedTrainer', LIKE_NNSSL_PLANS, '3d_fullres'), ('nnUNetTrainer', 'nnUNetPlans', '2d')]
        self.assertEqual(self._run(models, 0.8, 0.9), models)


class TestExportFindsEnsembles(unittest.TestCase):
    """nnUNetv2_export_model_to_zip must find the ensembles of the exported models, whatever their plans are called."""

    def _touch(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()

    def _make_model(self, results, trainer, plans, configuration):
        folder = os.path.join(results, 'Dataset004_X', f'{trainer}__{plans}__{configuration}')
        for f in ('plans.json', 'dataset.json', 'dataset_fingerprint.json', 'fold_0/checkpoint_final.safetensors',
                  'fold_0/progress.png', 'fold_0/validation/summary.json'):
            self._touch(os.path.join(folder, f))
        return folder

    def test_ensembles_with_double_underscore_plans(self):
        with TemporaryDirectory() as tmp:
            results = os.path.join(tmp, 'results')
            pt_2d = self._make_model(results, 'PretrainedTrainer', LIKE_NNSSL_PLANS, '2d')
            pt_3d = self._make_model(results, 'PretrainedTrainer', LIKE_NNSSL_PLANS, '3d_fullres')
            default_3d = self._make_model(results, 'nnUNetTrainer', 'nnUNetPlans', '3d_fullres')
            default_2d = self._make_model(results, 'nnUNetTrainer', 'nnUNetPlans', '2d')
            ensembles = os.path.join(results, 'Dataset004_X', 'ensembles')
            ours = get_ensemble_name(pt_2d, pt_3d, (0,))
            others = [get_ensemble_name(pt_3d, default_3d, (0,)), get_ensemble_name(default_2d, default_3d, (0,))]
            for e in [ours] + others:
                self._touch(os.path.join(ensembles, e, 'summary.json'))

            output_file = os.path.join(tmp, 'export.zip')
            with mock.patch.object(file_path_utilities, 'nnUNet_results', results), \
                    mock.patch.object(model_export_module, 'nnUNet_results', results):
                model_export_module.export_pretrained_model(
                    'Dataset004_X', output_file, configurations=('2d', '3d_fullres'), trainer='PretrainedTrainer',
                    plans_identifier=LIKE_NNSSL_PLANS, folds=(0,))
            with zipfile.ZipFile(output_file) as z:
                exported_ensembles = {n.split('/')[2] for n in z.namelist() if n.split('/')[1] == 'ensembles'}
            self.assertEqual(exported_ensembles, {ours})


if __name__ == '__main__':
    unittest.main()
