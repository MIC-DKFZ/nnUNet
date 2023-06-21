import yaml
import torch
import shutil
import umap

import numpy as np
import pandas as pd
import torch.nn as nn

from pathlib import Path
from sklearn import decomposition, manifold
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union, Dict, List
import captum.attr as attr
import SimpleITK as sitk
import multiprocessing as mp

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from ssl.models import nnunet_encoder, DINOHead, DynamicMultiCropWrapper
from ssl.ssl_utils import save_img_from_array_using_referece


class NetProjector(nn.Module):
    def __init__(self, cfg_file_path: Path, chckpt_file_path: Path,
                 device: str = 'cuda') -> None:
        """ Common 'network/projector' object used across all other classes.
        Args:
            cfg_file_path (Path): Path to the SSL configuration yaml file.
            chckpt_file_path (Path): PAth to the checkpoint.pth file to use.
            device (str, optional): Device in which to run the predictions.
                Defaults to 'cuda'.
        """
        super(NetProjector, self).__init__()

        # use the configuration file generated to train the SSL encoder
        with open(cfg_file_path, 'r') as yfile:
            self.cfg = yaml.safe_load(yfile)['training']

        # Define device and batch size
        self.device = device
        self.batch_size = 1  # self.cfg['batch_size'] # Hardcoded

        # instantiate the model
        self.teacher = nnunet_encoder(self.cfg, teacher=True, device=device)
        n_chnnel = 2 if self.cfg['multichannel_input'] else 1
        # print(n_chnnel)
        batch_shape = [self.batch_size, n_chnnel] + self.cfg['global_crop_size']
        self.projector = DynamicMultiCropWrapper(self.teacher, DINOHead, self.cfg['out_dim'],
                                                 self.cfg['use_bn_in_head'], batch_shape, True)

        # Turn off gradients
        for j, p in enumerate(self.projector.parameters()):
            p.requires_grad_(False)

        # Send to device and load retrained weights
        self.projector.to(self.device)
        state_dict = torch.load(chckpt_file_path, map_location=device)['teacher']
        self.projector.load_state_dict(state_dict, strict=True)

        # Set in evaluation mode
        self.projector.eval()


class EmpeddingProjector(NetProjector):
    def __init__(self, cfg_file_path: Path = None, chckpt_file_path: Path = None,
                 device: str = 'cuda') -> None:
        # Instatiate the net projector
        super(EmpeddingProjector, self).__init__(cfg_file_path, chckpt_file_path, device)

    def __call__(self, dataset: Union[Dataset, str]) -> pd.DataFrame:
        """Given a StrokeDataset object of the naem of the dataset,
        it generates the projections of the preprocessed images.
        Args:
            dataset (Union[Dataset, str]): Either the StrokeDataset object
                or a string with the name of the dataset to process. Be
                careful this string is not used, but iether the dataset 
                of the configuration file
        Returns:
            pd.Dataframe: Dataframe containin all the cases projections and
                some metadata columns: ['subject', 'ais', 'dataset_name'] 
                Every feature columns starts with 'feat' e.g. 'feat0'
        """

        # Define the preprocessed images path based on configuration file
        nnu_dataset_name = self.cfg['dataset']
        nnu_planner = self.cfg['exp_planner']
        nnu_configuration = self.cfg['configuration']
        preprocessed_dataset_path = \
            Path(nnUNet_preprocessed) / nnu_dataset_name / f'{nnu_planner}_{nnu_configuration}'
        
        # Get a list of the subjectects to process, you can pass either the StrokeDataset
        # object or use the images in the preprocessed path,
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

        # Once you have the list of images to process, run the model
        results = []
        for idx, npy_path in tqdm(enumerate(files), total=len(files)):
            # Load the file
            array = np.load(npy_path)
            # Check if the image has more than one channel
            if (len(array.shape) > 3) and (array.shape[0] == 2):
                # If so, check if we should keep only the first channel
                if self.cfg['transformations_cfg']['symmetry'] is not None:
                    array = np.expand_dims(array[0, ...], axis=0)
                elif not (self.cfg['multichannel_input']):
                    array = np.expand_dims(array[0, ...], axis=0)
            # Add the batch dimension
            array = np.expand_dims(array, axis=0)

            # Send array to device as tensor and process
            array = torch.tensor(array, device=self.device)
            with torch.no_grad():
                _, _, result = self.projector(array)

            # Keep all the projections
            results.append(
                [subjects[idx], ais[idx], dataset_names[idx]] + 
                result.T.detach().cpu().numpy().flatten().tolist()
            )

        # Generate the final pandas dataframe with the projections to be returned
        projection_size = len(results[0]) - 3
        columns = ['subject', 'ais', 'dataset_name'] + [f'feat{i}' for i in range(projection_size)]
        results = pd.DataFrame(results, columns=columns)
        return results


class DimReductor():
    def __init__(self, method: str = 'pca', seed: int = 420,
                 method_kwargs: Dict = {}) -> None:
        """Generates a dimensionality reduction model.
        Args:
            method (str, optional): Dimensionality reduction method to use.
                It can be either 'tse', 'umap', 'pca'. Defaults to 'pca'.
            seed (int, optional): Random seed to use in the dimensionality reduction.
                Defaults to 420.
            method_kwargs (Dict, optional): Any extra arguments to pass to the
                dimensionality reduction models check sklearn for deatils.
                Defaults to {}.
        """
        super(DimReductor, self).__init__()

        # Instantiate the dimentionality reduction model based on the given arguments
        self.method = method.lower()
        if self.method == 'pca':
            if (self.method in ['pca', 'incrementalpca']) and ('whiten' not in method_kwargs.keys()):
                method_kwargs['whiten'] = True
            self.model = decomposition.PCA(**method_kwargs)
        elif self.method == 'tsne':
            method_kwargs.update({'n_jobs': mp.cpu_count()})
            self.model = manifold.TSNE(**method_kwargs)
        elif self.method == 'umap':
            method_kwargs.update({'n_jobs': mp.cpu_count()})
            self.model = umap.UMAP(**method_kwargs)

        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, projections: Union[pd.DataFrame, np.ndarray],
                 n_to_fit: int = None) -> pd.DataFrame:
        """_summary_

        Args:
            projections (Union[pd.DataFrame, np.ndarray]): 
                The embedding generated by the neaural network encoder.
                It can be either an array or a pd.DataFrame. 
                Rows should be the exampels and columns the features.
            n_to_fit (int, optional): If too many cases are present in the
                set to reduce, then you can select a number of samples that
                will be randomly sampled and used to fit the model, then
                the fitted model is going to be used to project all the
                datapoints. Defaults to None, which means all samples are used.
        Returns:
            pd.DataFrame: Dataframe containing the projections of the
                samples given as input in the same order. If a dataframe
                pas given as input the non-feature columns are kept.
        """
        meta = None

        # IF df as input, get the features array
        if isinstance(projections, pd.DataFrame):
            feat_columns = [col for col in projections.columns if 'feat' in col]
            extra_cols = [col for col in projections.columns if 'feat' not in col]
            meta = projections[extra_cols]
            projections = projections[feat_columns].values

        # If not all samples should be used to fit the model
        if n_to_fit is not None:
            # sample a subset
            idxs = self.rng.random.randint(projections.shape[0], size=n_to_fit)
            sub_projections = projections[idxs, :]
            # fit the model
            self.model.fit(sub_projections)
            # project
            projections = self.model.transform(projections)
        else:
            # fit and project using all datapoints
            projections = self.model.fit_transform(projections)

        # generate a pd.Dataframe of the projections
        projections = pd.DataFrame(projections,
                                   columns=[f'feat{i}' for i in range(projections.shape[1])])

        #if a dataframe was pased, keep the non-feature/metadata columns
        if meta is not None:
            projections = pd.concat([meta, projections], axis=1)
        return projections


class AttrProjector(NetProjector):
    def __init__(self, cfg_file_path: Path, chckpt_file_path: Path = None,
                 device: str = 'cuda') -> None:
        """Attirbution Projector class. In order to get the atributtions of the model
        we need to add a las adding layer from all the elements in the bottleneck layer.
        Args:
            cfg_file_path (Path): Path to the configuration file of the SSL experiment.
            chckpt_file_path (Path): Path to the checkpoints to use in the model.
            device (str, optional): DEvice in which to run the predictions.
                Defaults to 'cuda'.
        """
        # Instantiate the network projector
        super(AttrProjector, self).__init__(cfg_file_path, chckpt_file_path, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, result = self.projector(x)
        return result.sum(dim=(1))


class AttrGenerator():
    def __init__(self, cfg_file_path: Path, chckpt_file_path: Path,
                 method: attr.Attribution, device: str = 'cuda', npy_path: Path = None,
                 raw_path: Path = None, layer: bool = False, suffix: str='') -> None:
        """A Saliency generator object that takes advantage of the SSL configuration file.
        Args:
            cfg_file_path (Path): Path to the SSL configuration file to use.
            chckpt_file_path (Path): Path to the heckpoint of the model to use.
            method (attr.Attribution): Method from captum.attr to use in the
                attribution maps generation, check captums documentation.
            device (str, optional): Device in which to tun the predictions.
                Defaults to 'cuda'.
            npy_path (Path, optional): Path to the preprocessed images.
                Defaults to None, which means it is obtained from the cfg file.
            raw_path (Path, optional): Path to the raw nii images.
                Defaults to None, which means it is obtained from the cfg file.
            layer (bool, optional): Some methods require a layer to be given.
                If the selected one is one of those, indicate it with True here.
                Defaults to False.
            suffix (str, optional): Suffix to use while saving the attribution maps.
                Defaults to ''.
        """

        # Instatiate the Atrributes projector
        self.saliency_generator = AttrProjector(cfg_file_path, chckpt_file_path, device)

        # Get the configuration file
        self.cfg = self.saliency_generator.cfg

        # Instantiate the captum attribution retrival object
        if layer:
            # If the attribution method requires a layer to be given
            # use a fixed one (first conv layer in the model)
            lay = self.saliency_generator.projector.backbone.net.stages[0][0].convs[1].all_modules[0]
            self.saliency_generator = method(self.saliency_generator,layer=lay)
        else:
            self.saliency_generator = method(self.saliency_generator)

        # Set other attibutes
        self.npy_path = npy_path
        self.device = device
        self.suffix = suffix

        # Fix a random number generator to always generate the same sampled cases
        self.rng = np.random.default_rng(420)

        # If no numpy path is given use the cfg file to get the preprocessed and raw paths
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


    def __call__(self, subjects: List[str] = [], n: int = None,
                 save_saliency: bool = True, out_path: Path = None,
                 copy_original: bool = False, return_arrays: bool = False
                 ) -> Union[None, List[np.ndarray]]:
        """Given a list of subjects, the saliency generator generates the attibution
        maps of all the cases or of the desired n random samples.
        """

        # If the output path doesn't exist, generate it
        out_path.mkdir(parents=True, exist_ok=True)

        # If only some cases are required, sample them
        n = len(subjects) if (n is None) else n
        if n != len(subjects):
            n_cases = len(subjects)
            indexes = self.rng.integers(0, n_cases, n)
            subjects = [self.dataset[idx]['subject'] for idx in indexes]

        # Get the paths to the images
        files = [self.preprocessed_dataset_path / f'{subj}.npy' for subj in subjects]

        # Adapt the suffix
        suffix = f'_{self.suffix}' if self.suffix != '' else self.suffix

        if return_arrays:
            # If the arrays are requested, accumulate them in a list
            results = []

        for npy_path in tqdm(files, total=len(files)):
            # Get the image and convert to tensor
            array = np.load(npy_path)
            n_channels = array.shape[0]
            if n_channels > 1 and not self.cfg['multichannel_input']:
                array = array[0, ...]
                array = np.expand_dims(array, axis=0)
                n_channels = 1
            array = np.expand_dims(array, axis=0)
            array = torch.tensor(array, device=self.device)

            # Get the attibution map
            result = self.saliency_generator.attribute(
                array, internal_batch_size=1).detach().cpu().numpy()
            # If the attribution maps need to be saved:
            if save_saliency:
                # Read the reference image
                nii_file = self.raw_path / npy_path.name.replace('.npy', '_0000.nii.gz')
                ncct_img = sitk.ReadImage(str(nii_file))
                # If the original image is requested to be saved nearby:
                if copy_original:
                    shutil.copy(nii_file, out_path/nii_file.name)

            for ch in range(n_channels):
                # If the attribution maps need to be saved:
                if save_saliency:                    
                    # Generate the output filepath and save
                    out_name = npy_path.name
                    if ch == 1:
                        out_name = out_name.replace('.npy', f'_saliency{suffix}_diff.nii.gz')
                    else:
                        out_name = out_name.replace('.npy', f'_saliency{suffix}.nii.gz')
                    out_filepath = out_path / out_name
                    save_img_from_array_using_referece(result[0, ch, :, :, :],
                                                       ncct_img, out_filepath)

                # If the arrays are requested, accumulate them in a list
                if return_arrays:
                    results.append(result)

            # Clear the cache to avoid excess of VRAM memmory consumption
            torch.cuda.empty_cache()

        if return_arrays:
            return results
