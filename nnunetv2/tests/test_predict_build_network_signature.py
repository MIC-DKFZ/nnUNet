import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class _DummyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_state = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.loaded_state = state_dict


class TestPredictBuildNetworkSignature(unittest.TestCase):
    def _run_initialize_with_trainer(self, trainer_cls):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
        )
        config_mgr = SimpleNamespace(
            network_arch_class_name='ResidualEncoderUNet',
            network_arch_init_kwargs={'n_stages': 6},
            network_arch_init_kwargs_req_import=('torch.nn',),
            patch_size=(128, 128, 128),
        )
        label_mgr = SimpleNamespace(num_segmentation_heads=3)
        plans_mgr = SimpleNamespace(
            get_configuration=lambda name: config_mgr,
            get_label_manager=lambda dj: label_mgr,
        )
        checkpoint = {
            'network_weights': {'w': 1},
            'trainer_name': trainer_cls.__name__,
            'init_args': {'configuration': '3d_fullres'},
            'inference_allowed_mirroring_axes': (0, 1, 2),
        }

        with (
            mock.patch('nnunetv2.inference.predict_from_raw_data.load_json', return_value={}),
            mock.patch('nnunetv2.inference.predict_from_raw_data.PlansManager', return_value=plans_mgr),
            mock.patch('nnunetv2.inference.predict_from_raw_data.load_checkpoint_file', return_value=checkpoint),
            mock.patch('nnunetv2.inference.predict_from_raw_data.determine_num_input_channels', return_value=1),
            mock.patch(
                'nnunetv2.inference.predict_from_raw_data.recursive_find_trainer_class_by_name',
                return_value=trainer_cls,
            ),
        ):
            predictor.initialize_from_trained_model_folder('/tmp/model', use_folds=(0,))

        return predictor

    def test_pretrained_trainer_signature_receives_input_patch_size_and_channels(self):
        captured = {}

        class PretrainedStyleTrainer:
            @staticmethod
            def build_network_architecture(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                input_patch_size,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=True,
                **kwargs,
            ):
                captured.update(
                    architecture_class_name=architecture_class_name,
                    arch_init_kwargs=arch_init_kwargs,
                    arch_init_kwargs_req_import=arch_init_kwargs_req_import,
                    input_patch_size=input_patch_size,
                    num_input_channels=num_input_channels,
                    num_output_channels=num_output_channels,
                    enable_deep_supervision=enable_deep_supervision,
                )
                return _DummyNetwork()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DeprecationWarning)
            predictor = self._run_initialize_with_trainer(PretrainedStyleTrainer)

        self.assertIsInstance(predictor.network, _DummyNetwork)
        self.assertEqual(captured['architecture_class_name'], 'ResidualEncoderUNet')
        self.assertEqual(captured['input_patch_size'], (128, 128, 128))
        self.assertEqual(captured['num_input_channels'], 1)
        self.assertEqual(captured['num_output_channels'], 3)
        self.assertFalse(captured['enable_deep_supervision'])
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_primusx_trainer_signature_receives_input_patch_size(self):
        captured = {}

        class PrimusXStyleTrainer:
            @staticmethod
            def build_network_architecture(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=True,
                input_patch_size=None,
            ):
                captured.update(
                    input_patch_size=input_patch_size,
                    num_input_channels=num_input_channels,
                    num_output_channels=num_output_channels,
                    enable_deep_supervision=enable_deep_supervision,
                )
                return _DummyNetwork()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', DeprecationWarning)
            self._run_initialize_with_trainer(PrimusXStyleTrainer)

        self.assertEqual(captured['input_patch_size'], (128, 128, 128))
        self.assertEqual(captured['num_input_channels'], 1)
        self.assertEqual(captured['num_output_channels'], 3)
        self.assertFalse(captured['enable_deep_supervision'])

    def test_legacy_trainer_without_input_patch_size_remains_supported(self):
        captured = {}

        class LegacyStyleTrainer:
            @staticmethod
            def build_network_architecture(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=True,
            ):
                captured.update(
                    architecture_class_name=architecture_class_name,
                    num_input_channels=num_input_channels,
                    num_output_channels=num_output_channels,
                    enable_deep_supervision=enable_deep_supervision,
                )
                return _DummyNetwork()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always', DeprecationWarning)
            self._run_initialize_with_trainer(LegacyStyleTrainer)

        self.assertEqual(captured['architecture_class_name'], 'ResidualEncoderUNet')
        self.assertEqual(captured['num_input_channels'], 1)
        self.assertEqual(captured['num_output_channels'], 3)
        self.assertFalse(captured['enable_deep_supervision'])


if __name__ == '__main__':
    unittest.main()
