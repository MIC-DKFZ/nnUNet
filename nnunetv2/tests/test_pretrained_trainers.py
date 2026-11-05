import io
import os
import tempfile
import unittest
import warnings
from unittest import mock

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json

from nnunetv2.experiment_planning import like_nnssl
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.pretraining.dynamicPretrainedTrainer import DynamicPretrainedTrainer
from nnunetv2.utilities.checkpoints import save_checkpoint_file
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

PLAIN_UNET = {
    'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
    'arch_kwargs': {
        'n_stages': 3, 'features_per_stage': [4, 8, 16], 'conv_op': 'torch.nn.modules.conv.Conv3d',
        'kernel_sizes': [[3, 3, 3]] * 3, 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]], 'n_conv_per_stage': [1, 1, 1],
        'n_conv_per_stage_decoder': [1, 1], 'conv_bias': True,
        'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True},
    },
    '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
}

RESENC_UNET = {
    'network_class_name': 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
    'arch_kwargs': {
        'n_stages': 3, 'features_per_stage': [4, 8, 16], 'conv_op': 'torch.nn.modules.conv.Conv3d',
        'kernel_sizes': [[3, 3, 3]] * 3, 'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]], 'n_blocks_per_stage': [1, 2, 2],
        'n_conv_per_stage_decoder': [1, 1], 'conv_bias': True,
        'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True},
    },
    '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
}

PRIMUSX = {
    'network_class_name': 'PrimusX',
    'arch_kwargs': {'embed_dim': 96, 'patch_embed_size': [8, 8, 8], 'encoder_eva_depth': 2, 'encoder_eva_numheads': 4,
                    'scale_attn_inner': True, 'init_values': 0.1},
    '_kw_requires_import': [],
}


def _plain_unet(patch_size):
    a = PLAIN_UNET
    return get_network_from_plans(a['network_class_name'], a['arch_kwargs'], a['_kw_requires_import'], 1, 3,
                                  allow_init=True, deep_supervision=False)


def _resenc_unet(patch_size):
    a = RESENC_UNET
    return get_network_from_plans(a['network_class_name'], a['arch_kwargs'], a['_kw_requires_import'], 1, 3,
                                  allow_init=True, deep_supervision=False)


def _primusx(patch_size):
    from dynamic_network_architectures.architectures.primus import Primus
    k = PRIMUSX['arch_kwargs']
    return Primus(1, k['embed_dim'], k['patch_embed_size'], 3, k['encoder_eva_depth'], k['encoder_eva_numheads'],
                  patch_size, drop_path_rate=0.2, scale_attn_inner=k['scale_attn_inner'], init_values=k['init_values'])


class TestPretrainedTrainersAtInference(unittest.TestCase):
    """
    nnUNetv2_predict rebuilds the network through the trainer's build_network_architecture. The finetuning trainers
    must build the same network there as during training, including the patch size Primus needs.
    """

    def check(self, trainer_name: str, architecture: dict, patch_size, build_reference):
        torch.manual_seed(0)
        reference = build_reference(patch_size)
        with tempfile.TemporaryDirectory() as model:
            plans = {'plans_name': 'ptPlans', 'configurations': {'3d_fullres': {
                'patch_size': patch_size, 'architecture': architecture}}}
            save_json(plans, join(model, 'plans.json'))
            save_json({'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'a': 1, 'b': 2},
                       'file_ending': '.nii.gz'}, join(model, 'dataset.json'))
            maybe_mkdir_p(join(model, 'fold_0'))
            save_checkpoint_file({'network_weights': reference.state_dict(), 'trainer_name': trainer_name,
                                  'init_args': {'configuration': '3d_fullres'},
                                  'inference_allowed_mirroring_axes': (0, 1, 2)},
                                 join(model, 'fold_0', 'checkpoint_final.safetensors'))

            predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
            with mock.patch.dict(os.environ, {'nnUNet_compile': 'f'}), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                predictor.initialize_from_trained_model_folder(model, use_folds=(0,))
        self.assertFalse([w for w in caught if 'old build_network_architecture signature' in str(w.message)])

        x = torch.rand(1, 1, *patch_size)
        reference.eval()
        predictor.network.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(reference(x), predictor.network(x)))

    def test_pretrained_trainer(self):
        self.check('PretrainedTrainer', PLAIN_UNET, [16, 32, 32], _plain_unet)

    def test_dynamic_pretrained_trainer(self):
        self.check('DynamicPretrainedTrainer', PLAIN_UNET, [16, 32, 32], _plain_unet)

    def test_primusx_trainer(self):
        self.check('PretrainedTrainer_Primusx', PRIMUSX, [32, 48, 32], _primusx)



class TestDynamicPretrainedWeightLoading(unittest.TestCase):
    """
    DynamicPretrainedTrainer matches the checkpoint's encoder weights to the network by parameter name and skips
    everything else. A checkpoint of another architecture (a ResEnc checkpoint with the PlainConvUNet of nnUNetPlans)
    used to load nothing at all, silently, and finetuning ran from random weights (GitHub issue #3085).
    """

    @staticmethod
    def load(pretrained, target, patch_size=(16, 32, 32)):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = join(tmp, 'checkpoint.pth')
            plan = {'key_to_encoder': pretrained.key_to_encoder, 'key_to_stem': pretrained.key_to_stem,
                    'keys_to_in_proj': list(pretrained.keys_to_in_proj), 'key_to_lpe': None,
                    'pretrain_patch_size': list(patch_size),
                    'architecture_kwargs': {'n_stages': 3, 'kernel_sizes': [[3, 3, 3]] * 3}}
            torch.save({'network_weights': pretrained.state_dict(), 'nnssl_adaptation_plan': plan}, ckpt)
            with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                DynamicPretrainedTrainer.load_pretrained_weights_dynamic(
                    target, ckpt, 1, 1, list(patch_size), list(patch_size), plan['key_to_encoder'],
                    plan['key_to_stem'], tuple(plan['keys_to_in_proj']), None, [[3, 3, 3]] * 3, 3)
        return stdout.getvalue()

    def test_resenc_checkpoint_into_plainconv_raises(self):
        torch.manual_seed(0)
        with self.assertRaisesRegex(RuntimeError, 'None of the pretrained encoder weights fit'):
            self.load(_resenc_unet(None), _plain_unet(None))

    def test_resenc_checkpoint_into_resenc_loads_everything(self):
        torch.manual_seed(0)
        pretrained = _resenc_unet(None)
        torch.manual_seed(1)
        target = _resenc_unet(None)
        output = self.load(pretrained, target)
        self.assertIn('100.0 %', output)
        target_encoder = dict(target.encoder.named_parameters())
        for name, p in pretrained.encoder.named_parameters():
            self.assertTrue(torch.equal(p, target_encoder[name]), name)


class TestPlanLikeDynamicArchitectureCheck(unittest.TestCase):
    RESENC = 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
    PLAIN = 'dynamic_network_architectures.architectures.unet.PlainConvUNet'

    def test_matching_and_unknown_architectures_pass(self):
        for pt, base in [('ResEncL', self.RESENC), ('NoSkipResEncL', self.RESENC), ('ResidualEncoderUNet', self.RESENC),
                         (self.RESENC, self.RESENC), ('PlainConvUNet', self.PLAIN), (None, self.PLAIN),
                         ('SomethingNew', self.PLAIN), ('ResEncL', 'my_package.MyUNet')]:
            like_nnssl.check_base_plan_matches_checkpoint(base, pt, 'somePlans')

    def test_other_architecture_family_raises(self):
        for pt, base in [('ResEncL', self.PLAIN), ('ResidualEncoderUNet', self.PLAIN), ('PlainConvUNet', self.RESENC),
                         ('PrimusM', self.RESENC)]:
            with self.assertRaisesRegex(ValueError, 'does not match the architecture of the base plan'):
                like_nnssl.check_base_plan_matches_checkpoint(base, pt, 'somePlans')

    def test_plan_like_dynamic_refuses_before_writing_anything(self):
        dataset_name = 'Dataset987_ArchCheck'
        with tempfile.TemporaryDirectory() as tmp:
            maybe_mkdir_p(join(tmp, dataset_name))
            save_json({'plans_name': 'nnUNetPlans', 'configurations': {'3d_fullres': {
                'patch_size': [16, 32, 32], 'architecture': PLAIN_UNET}}}, join(tmp, dataset_name, 'nnUNetPlans.json'))
            ckpt = join(tmp, 'checkpoint.pth')
            torch.save({'nnssl_adaptation_plan': {'architecture_plans': {'arch_class_name': 'ResEncL'}}}, ckpt)
            with mock.patch.object(like_nnssl, 'nnUNet_preprocessed', tmp), \
                    mock.patch.object(like_nnssl, 'convert_id_to_dataset_name', return_value=dataset_name):
                with self.assertRaisesRegex(ValueError, 'does not match the architecture of the base plan'):
                    like_nnssl.plan_like_dynamic(987, 'test', ckpt, 'nnUNetPlans')
            self.assertEqual(os.listdir(join(tmp, dataset_name)), ['nnUNetPlans.json'])


if __name__ == '__main__':
    unittest.main()
