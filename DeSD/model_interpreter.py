import yaml
import torch
import shutil

import numpy as np
import pandas as pd
import torch.nn as nn

from pathlib import Path
from sklearn import decomposition
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union, Dict, List
import captum.attr as attr
import SimpleITK as sitk

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from DeSD.models.res3d import res3d, DINOHead, DynamicMultiCropWrapper
from DeSD.utils_desd import save_img_from_array_using_referece


class NetProjector(nn.Module):
    def __init__(
        self, cfg_file_path: Path, chckpt_file_path: Path, device: str = 'cuda', batch_size: int = None
    ):
        super(NetProjector, self).__init__()

        with open(cfg_file_path, 'r') as yfile:
            self.cfg = yaml.safe_load(yfile)['training']
        self.device = device
        self.batch_size = batch_size if batch_size is not None else self.cfg['batch_size']
        self.teacher = res3d(self.cfg, teacher=True, device=device)
        batch_shape = [self.batch_size, 1] + self.cfg['global_crop_size']
        self.projector = DynamicMultiCropWrapper(self.teacher, DINOHead, self.cfg['out_dim'],
                                                 self.cfg['use_bn_in_head'], batch_shape, True)
        self.projector.to(self.device)
        state_dict = torch.load(chckpt_file_path, map_location=device)['teacher']
        self.projector.load_state_dict(state_dict, strict=True)
        self.projector.eval()


class EmpeddingProjector(NetProjector):
    def __init__(self, net_projector: NetProjector = None, cfg_file_path: Path = None,
                 chckpt_file_path: Path = None, device: str = 'cuda', batch_size: int = None):
        # if not ((net_projector is None) and ((cfg_file_path is not None) and (chckpt_file_path is not None))):
        #     raise Exception('You either need to pass the net projector or the cfg and checkpt paths')
        # if not ((net_projector is not None) and ((cfg_file_path is None) and (chckpt_file_path is None))):
        #     raise Exception('You either need to pass the net projector or the cfg and checkpt paths')
        if net_projector is None:
            super(EmpeddingProjector, self).__init__(cfg_file_path, chckpt_file_path, device, batch_size)
        else:
            super(EmpeddingProjector, self).__init__()
            self.cfg = net_projector.cfg
            self.batch_size = net_projector.batch_size
            self.teacher = net_projector.teacher
            self.projector = net_projector.projector
            self.device = device

    def __call__(self, dataset: Union[Dataset, str]):
        nnu_dataset_name = self.cfg['dataset']
        nnu_planner = self.cfg['exp_planner']
        nnu_configuration = self.cfg['configuration']
        preprocessed_dataset_path = \
            Path(nnUNet_preprocessed) / nnu_dataset_name / f'{nnu_planner}_{nnu_configuration}'
        if isinstance(dataset, str):
            files = [i for i in preprocessed_dataset_path.iterdir() if i.name.endswith('.npy')]
            subjects = [i.name.rstrip('.npy') for i in files]
            dataset_names = ais = ['-'] *  len(files)
        elif isinstance(dataset, Dataset):
            dataset_names, files, ais, subjects = [], [], [], []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                subjects.append(sample['subject'])
                dataset_names.append(sample['dataset_name'])
                ais.append(sample['ais'])
                files.append(preprocessed_dataset_path / f'{sample["subject"]}.npy')

        results = []
        for idx, npy_path in tqdm(enumerate(files)):
            array = np.load(npy_path)
            array = np.expand_dims(array, axis=0)
            array = torch.tensor(array, device=self.device)
            with torch.no_grad():
                _, _, result = self.projector(array)
            results.append(
                [subjects[idx], ais[idx], dataset_names[idx]] + 
                result.T.detach().cpu().numpy().flatten().tolist()
            )
        projection_size = len(results[0]) - 3
        columns = ['subject', 'ais', 'dataset_name'] + [f'feat{i}' for i in range(projection_size)]
        results = pd.DataFrame(results, columns=columns)
        return results


class DimReductor():
    def __init__(
        self, method: str = 'pca', verbose: bool = False,
        seed: int = 420, method_kwargs: Dict = {}
    ):
        super(DimReductor, self).__init__()
        self.method = method.lower()

        if (self.method in ['pca', 'incrementalpca']) and ('whiten' not in method_kwargs.keys()):
            method_kwargs['whiten'] = True
        self.model = decomposition.PCA(**method_kwargs)
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, projections: Union[pd.DataFrame, np.ndarray], n_to_fit: int = None):
        meta = None
        if isinstance(projections, pd.DataFrame):
            feat_columns = [col for col in projections.columns if 'feat' in col]
            extra_cols = [col for col in projections.columns if 'feat' not in col]
            meta = projections[extra_cols]
            projections = projections[feat_columns].values

        if n_to_fit is not None:
            idxs = self.rng.random.randint(projections.shape[0], size=n_to_fit)
            sub_projections = projections[idxs, :]
            self.model.fit(sub_projections)
            projections = self.model.transform(projections)
        else:
            projections = self.model.fit_transform(projections)

        projections = pd.DataFrame(projections,
                                   columns=[f'{self.method}{i}' for i in range(projections.shape[1])])
        if meta is not None:
            projections = pd.concat([meta, projections], axis=1)
        return projections


class CAMProjector(NetProjector):
    def __init__(self, net_projector: NetProjector = None, cfg_file_path: Path = None,
                 chckpt_file_path: Path = None, device: str = 'cuda', batch_size: int = None):
        # if not ((net_projector is None) and ((cfg_file_path is not None) and (chckpt_file_path is not None))):
        #     raise Exception('You either need to pass the net projector or the cfg and checkpt paths')
        # if not ((net_projector is not None) and ((cfg_file_path is None) and (chckpt_file_path is None))):
        #     raise Exception('You either need to pass the net projector or the cfg and checkpt paths')
        if net_projector is None:
            super(CAMProjector, self).__init__(cfg_file_path, chckpt_file_path, device, batch_size)
        else:
            self.cfg = net_projector.cfg
            self.batch_size = net_projector.batch_size
            self.teacher = net_projector.teacher
            self.projector = net_projector.projector

    def forward(self, x):
        _, _, result = self.projector(x)
        return result.sum(dim=(1))


class SaliencyGenerator():
    def __init__(self,
                 net_projector: NetProjector = None,
                 cfg_file_path: Path = None,
                 chckpt_file_path: Path = None,
                 device: str = 'cuda',
                 dataset: Dataset = None,
                 npy_path: Path = None,
                 raw_path: Path = None,
                 method: attr.Attribution = None, method_kwargs: Dict = {}, layer: bool = False, suffix: str=''):
        # assert (net_projector is None) and ((cfg_file_path is not None) and (chckpt_file_path is not None)), \
        #     'You either need to pass the net projector or the cfg and checkpt paths'
        # assert (net_projector is not None) and ((cfg_file_path is None) and (chckpt_file_path is None)), \
        #     'You either need to pass the net projector or the cfg and checkpt paths'
        if net_projector is None:
            self.saliency_generator = CAMProjector(None, cfg_file_path, chckpt_file_path, device, 1)
        # else:
        #     self.saliency_generator = CAMProjector(net_projector)
        self.cfg = self.saliency_generator.cfg
        if layer:
            self.saliency_generator = method(self.saliency_generator, layer=self.saliency_generator.projector.backbone.net.stages[0][0].convs[1].all_modules[0])
        else:
            self.saliency_generator = method(self.saliency_generator)
        self.dataset = dataset
        self.npy_path = npy_path
        self.device = device
        self.suffix = suffix
        self.rng = np.random.default_rng(420)
        if npy_path is None:
            nnu_dataset_name = self.cfg['dataset']
            nnu_planner = self.cfg['exp_planner']
            nnu_configuration = self.cfg['configuration']
            self.preprocessed_dataset_path = \
                Path(nnUNet_preprocessed) / nnu_dataset_name / f'{nnu_planner}_{nnu_configuration}'
            self.raw_path = Path(nnUNet_raw) / nnu_dataset_name / f'imagesTr'
        else:
            self.preprocessed_dataset_path = npy_path
            self.raw_path = raw_path


    def __call__(self, subjects: List = [], n: int = None,
                 save_saliency: bool = True, out_path: Path = None):
        assert (len(subjects) == 0) and ((n != None) and (self.dataset is not None)), \
            'You must pass a list of cases or a number of cases to sample and a dataset object'
        if len(subjects) == 0:
            n_cases = len(self.dataset)
            indexes = self.rng.integers(0, n_cases, n)
            subjects = [self.dataset[idx]['subject'] for idx in indexes]
        files = [self.preprocessed_dataset_path / f'{subj}.npy' for subj in subjects]

        suffix = f'_{self.suffix}' if self.suffix != '' else self.suffix
        results = []
        for idx, npy_path in enumerate(files):
            array = np.load(npy_path)
            array = np.expand_dims(array, axis=0)
            array = torch.tensor(array, device=self.device)
            results.append(self.saliency_generator.attribute(array)[0, 0, :, :, :].detach().numpy())
            if save_saliency:
                nii_file = self.raw_path / npy_path.name.replace('.npy', '_0000.nii.gz')
                ncct_img = sitk.ReadImage(str(nii_file))
                shutil.copy(nii_file, out_path/nii_file.name)
                out_filepath = out_path / npy_path.name.replace('.npy', f'_saliency{suffix}.nii.gz')
                save_img_from_array_using_referece(results[-1], ncct_img, out_filepath)
        return results


