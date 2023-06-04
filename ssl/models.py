import torch
import shutil
import torch.nn as nn
from typing import Union
from pathlib import Path
from ssl.ssl_utils import trunc_normal_
from nnunetv2.run.run_training import get_trainer_from_args


def get_nnunet_backbone(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: int,
    trainer_class_name: str = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    device: torch.device = torch.device('cuda')
):
    
    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id, configuration, fold, trainer_class_name,
        plans_identifier, False, device=device)
    nnunet_trainer.initialize()
    Path(nnunet_trainer.log_file).unlink()
    shutil.rmtree(nnunet_trainer.tb_log_file)
    return nnunet_trainer.network.encoder


class nnunet_encoder(nn.Module):
    def __init__(self, cfg: dict, teacher: bool = False, device: str = None):
        super(nnunet_encoder, self).__init__()
        self.device = device if device is not None else cfg['device']
        self.teacher = teacher
        self.net = get_nnunet_backbone(cfg['dataset'], cfg['configuration'], cfg['fold'],
                                       cfg['trainer'], cfg['exp_planner'],
                                       torch.device(self.device))
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.net(x)
        out = [self.gap(x_i).flatten(1) for x_i in x]
        if self.teacher:
            return out[-1]
        return out


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        out = self.last_layer(x)
        return out, x


class PIXELHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Conv3d(in_dim, out_dim, kernel_size=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.mlp(x)
        return x


class DynamicMultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(
        self, backbone, head_class, out_dim, use_bn_in_head, batch_size, teacher=False
    ):
        super(DynamicMultiCropWrapper, self).__init__()
        random_tensor = torch.rand(batch_size).to(device=backbone.device)

        with torch.no_grad():
            temp = backbone.eval()
            output = temp(random_tensor)
            del temp
        self.n_heads = len([n for n in backbone.net.stages.children()]) if not teacher else 1
        
        if teacher:
            self.latent_sizes = [output[0].size()[0]]
        else:
            self.latent_sizes = [output[i].size()[1] for i in range(self.n_heads)]
        heads = [head_class(lsz, out_dim, use_bn_in_head) for lsz in self.latent_sizes]

        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.teacher = teacher
        self.head = heads[-1]
        if not self.teacher:
            for i in range(self.n_heads - 1):
                setattr(self, f'head_{i+1}', heads[i])
                setattr(self.backbone, f'head_{i+1}', nn.Identity()) 

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0 
        for end_idx in idx_crops:
            if self.teacher:
                _outs = self.backbone(torch.cat(x[start_idx: end_idx]))
                outputs = [_outs if start_idx == 0 else torch.cat((outputs, _outs))]
                start_idx = end_idx
            else:
                _outs = list(self.backbone(torch.cat(x[start_idx: end_idx])))
                outputs = _outs if start_idx == 0 else \
                    [torch.cat((outputs[i], _outs[i])) for i in range(len(_outs))]
                start_idx = end_idx
        # Run the head forward on the concatenated features.
        result = []
        for i in range(self.n_heads - 1):
            result.append(getattr(self, f'head_{i+1}')(outputs[i])[0])
        output, embeding = self.head(outputs[-1])   # respecting nomenclature from RankMe paper page 13
        result.append(output)
        return result, embeding, outputs[-1]