import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import chain
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import time
from typing import List, Dict, Tuple
from torch.utils.tensorboard import SummaryWriter


def get_latent_representations(student: nn.Module, teacher: nn.Module, data_loader: DataLoader,
                               n_samples: int, fp16_scaler: GradScaler, desd_loss: nn.Module,
                               epoch: int) -> Tuple[pd.DataFrame, Dict]:
    embedings, latent_representations = [], []
    average_loss = {"total_loss": [], "loss_ssl": []}
    for i in range(student.n_heads - 1):
        average_loss[f'deep_loss_{i+1}'] = []

    labels, dataset_names = [], []
    for k, images in enumerate(data_loader):
        # get as many train samples as samples are in the validation set
        if (k * data_loader.batch_size) >= n_samples:
            break

        labels.extend(images[1][0].tolist())
        dataset_names.extend(images[2][0])
        images = images[0]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get the latent_representations
            teacher_output, t_lat_rep, embeding = teacher(images[:2])
            student_output, _, _ = student(images)

            # get the loss
            losses = desd_loss(student_output, teacher_output, epoch)
            average_loss["total_loss"].append(losses[0].item())
            average_loss["loss_ssl"].append(losses[1].item())
            for i in range(student.n_heads - 1):
                average_loss[f'deep_loss_{i+1}'].append(losses[i+2].item())

            latent_representations.append(t_lat_rep.detach().chunk(2)[0].cpu().numpy())
            embedings.append(embeding.detach().chunk(2)[0].cpu().numpy())
    latent_representations = np.vstack(latent_representations)
    n_feats = latent_representations.shape[1]
    latent_representations = pd.DataFrame(latent_representations,
                                           columns=[f'feat_{i}' for i in range(n_feats)])
    latent_representations['ais'] = labels
    latent_representations['dataset_name'] = dataset_names

    embedings = np.vstack(embedings)
    n_feats = embedings.shape[1]
    embedings = pd.DataFrame(embedings, columns=[f'feat_{i}' for i in range(n_feats)])
    embedings['ais'] = labels
    embedings['dataset_name'] = dataset_names

    mean_losses = {k: np.mean(meter) for k, meter in average_loss.items()}
    return latent_representations, embedings, mean_losses

def get_rank_me(singular_vals: np.ndarray) -> float:
    norm_one = np.linalg.norm(singular_vals, 1)
    singular_vals = (singular_vals / norm_one) + 1e-7
    singular_vals_entropy = - np.sum(singular_vals * np.log(singular_vals))
    rank_me = np.exp(singular_vals_entropy)
    return rank_me

def logger(partition: str, log_dict: Dict, epoch: int, writer: SummaryWriter):
    partition = 'trn' if partition == 'train' else 'val'
    for k, v in log_dict.items():
        writer.add_scalar(f'{partition}_{k}', v, epoch)
    # logging
    rounded = {k: round(v, 4) for k, v in log_dict.items()}
    print(f"Epoch: {epoch}  -  Epoch end stats: {rounded}")

def log_scatterplot_figure(proj_df: pd.DataFrame, writer: SummaryWriter,
                           epoch: int, partition: str, name: str) -> None:

    proj_df['dataset_name'] = [i.upper() for i in proj_df['dataset_name'].tolist()]
    partition = 'validation' if (partition == 'val') else partition
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    palette = {True:'C0', False: 'C1'}
    sns.scatterplot(data= proj_df, x='pca_0', y='pca_1', hue='ais',
                    palette=palette, ax=ax[0])
    ax[0].set_title(f'PCA projection of {partition} set ({name})\nablated by stroke presence')
    ax[0].set_xlabel('First component')
    ax[0].set_ylabel('Second component')
    ax[0].legend(title='Stroke')
    sns.despine()

    palette = {'TUM':'C0', 'AISD': 'C1', 'APIS': 'C2', 'TBI': 'C3'}
    sns.scatterplot(data= proj_df, x='pca_0', y='pca_1', hue='dataset_name',
                    palette=palette, ax=ax[1])
    ax[1].set_title(f'PCA projection of {partition} set {name}\nablated by dataset origin')
    ax[1].set_xlabel('First component')
    ax[1].set_ylabel('Second component')
    ax[1].legend(title='Dataset')
    sns.despine()
    
    writer.add_figure(f'pca_{partition}', fig, global_step=epoch)


def log_example_images(images: List[np.ndarray], writer: SummaryWriter,
                       epoch: int) -> None:
    rng = np.random.default_rng(420)
    images = np.asarray([i.numpy() for i in images])
    images = np.concatenate(images, 4)
    idx = rng.choice(np.arange(images.shape[0]), 5)
    z_idx = images.shape[2] // 2
    n_channels = images.shape[1]
    imgs = (np.concatenate(images[idx, 0, z_idx, :, :], 0)).T
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.imshow(imgs, cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    writer.add_figure(f'exp_imgs', fig, global_step=epoch)
    if n_channels == 2:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        imgs = (np.concatenate(images[idx, 1, z_idx, :, :], 0)).T
        ax.imshow(imgs, cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        writer.add_figure(f'exp_imgs_ch2', fig, global_step=epoch)


@torch.no_grad()
def at_epochs_end(student: nn.Module, teacher: nn.Module, desd_loss: nn.Module,
                  train_data_loader: DataLoader, val_data_loader: DataLoader,
                  epoch: int, fp16_scaler: GradScaler, cfg: Dict,
                  writer: SummaryWriter, best_rankme: float) -> None:
    
    output_dir = Path(cfg['output_dir'])

    log_projections = (epoch == 0) or ((epoch+1) % cfg['projection_freq'] == 0)
    log_rankme = (epoch % cfg['rank_me_freq'] == 0)

    student.eval()
    teacher.eval()
    validation_samples = len(val_data_loader.dataset)
    if validation_samples > (3 * teacher.latent_sizes[-1]):
        validation_samples = 3 * teacher.latent_sizes[-1]

    for partition in cfg['over']:
        data_loader = train_data_loader if partition == 'train' else val_data_loader
        data_loader.rng = np.random.default_rng(420+epoch)
        # Get projections on train samples:
        lat_rep_df, embed_df, log_dict = get_latent_representations(
            student, teacher, data_loader, validation_samples, fp16_scaler, desd_loss, epoch)
        for df, name in zip([embed_df, lat_rep_df], ['bottleneck', 'head']):
            feat_cols = [i for i in df.columns if 'feat' in i]
            if log_rankme:
                np_rank = np.linalg.matrix_rank(df[feat_cols].values.astype('float32'))
            if log_rankme or log_projections:
                model = PCA(n_components=None, whiten=False, svd_solver='full', random_state=420)
                
                # project latent representations
                pca_projection = model.fit_transform(df[feat_cols].values)[:, :2]
                pca_projection_df = df.drop(columns=feat_cols)
                pca_projection_df.loc[:, ['pca_0', 'pca_1']] = pca_projection
                del df, pca_projection
                if log_projections:
                    log_scatterplot_figure(pca_projection_df, writer, epoch, partition, name)

                # Get rank measure to comapre it to jax_one
                if log_rankme:
                    pca_rank_me = get_rank_me(model.singular_values_)
                    if (name == 'head') and (partition != 'train') and (pca_rank_me > best_rankme) and (epoch > 5):
                        best_rankme = pca_rank_me
                        save_dict = {'student': student.state_dict(),
                                    'teacher': teacher.state_dict(),
                                    'epoch': epoch + 1,
                                    'args': cfg,
                                    'desd_loss': desd_loss.state_dict()}
                        if fp16_scaler is not None:
                            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                        torch.save(save_dict, str(output_dir/'checkpoint_rankme.pth'))
                        print('\nYay new rankme model saved\n')
                    log_dict.update({f'{name}_pca_rank_me': pca_rank_me, f'{name}_np_rank': np_rank})
        logger(partition, log_dict, epoch, writer)
    student.train()
    teacher.train()
    return best_rankme