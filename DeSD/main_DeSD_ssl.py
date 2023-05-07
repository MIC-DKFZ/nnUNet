import argparse
import datetime
import json
import math
import sys
import time
import torch
import yaml

import numpy as np
import torch.backends.cudnn as cudnn
import DeSD.utils_desd as utils

from pathlib import Path
from DeSD.desd_loss import DeSDLoss
from DeSD.models.res3d import DINOHead, res3d, DynamicMultiCropWrapper
from DeSD.data_loader_ssl import Dataset3D
from DeSD.utils_desd import copy_code, adapt_chckpt
from DeSD.metrics_utils import at_epochs_end, log_example_images

from nnunetv2.paths import nnUNet_preprocessed
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

datapath = Path('/home/jseia/Desktop/thesis/data/')


def train_DeSD(cfg: dict):
    # Guarantee reproducibility
    utils.fix_random_seeds(cfg['seed'])
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False

    output_dir = Path(cfg['output_dir'])
    cfg['metrics_cfg'].update({'output_dir': cfg['output_dir']})
    # Print general configuration of the training
    # print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(cfg.items())))

    # copy code files files for reproducibility
    copy_code(output_dir)

    # ============ preparing training data ... ============
    train_set = Dataset3D(cfg['dataset'], train=True, cfg = cfg)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=cfg['batch_size'],
                                                    num_workers=cfg['num_workers'],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    shuffle=True)

    print(f"Data loaded: there are {len(train_set)} images in the training set.")
    # ============ preparing validation data ... ============
    val_set = Dataset3D(cfg['dataset'], train=False, cfg = cfg)
    val_data_loader = torch.utils.data.DataLoader(val_set,
                                                  batch_size=cfg['batch_size'],
                                                  num_workers=cfg['num_workers'],
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  shuffle=True)
    print(f"Data loaded: there are {len(train_set)} images in the training set.")
    
    # ============ Create student and teacher models ... ============
    # apply_sym = cfg['transformations_cfg']['symmetry']
    channels = 2 if cfg['multichannel_input'] else 1

    batch_shape = [cfg['batch_size'], channels] + cfg['global_crop_size']
    student, teacher = res3d(cfg, teacher=False), res3d(cfg, teacher=True)
    student = DynamicMultiCropWrapper(student, DINOHead, cfg['out_dim'],
                                      cfg['use_bn_in_head'], batch_shape, False)
    teacher = DynamicMultiCropWrapper(teacher, DINOHead, cfg['out_dim'],
                                      cfg['use_bn_in_head'], batch_shape, True)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    teacher.load_state_dict(student.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f'Student and Teacher are built based on {cfg["exp_planner"]} file.')

    # ============ preparing loss ... ============
    desd_loss = DeSDLoss(out_dim=cfg['out_dim'],
                         ncrops=cfg['local_crops_number'] + 2,
                         n_heads=student.n_heads, 
                         warmup_teacher_temp=cfg['warmup_teacher_temp'],
                         teacher_temp=cfg['teacher_temp'],
                         warmup_teacher_temp_epochs=cfg['warmup_teacher_temp_epochs'],
                         nepochs=cfg['epochs'],
                         weights=cfg['weights']).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if cfg['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif cfg['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif cfg['optimizer'] == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if cfg['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(cfg['lr'] * cfg['batch_size'] / cfg['lr_sch_den'],  # linear scaling rule
                                         cfg['min_lr'],
                                         cfg['epochs'],
                                         len(train_data_loader),
                                         warmup_epochs=cfg['warmup_epochs'])
    wd_schedule = utils.cosine_scheduler(cfg['weight_decay'],
                                         cfg['weight_decay_end'],
                                         cfg['epochs'],
                                         len(train_data_loader))
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(cfg['momentum_teacher'],
                                               1,
                                               cfg['epochs'],
                                               len(train_data_loader))
    print("Loss, optimizer and schedulers ready.")
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(str(output_dir/'checkpoint.pth'),
                                  run_variables=to_restore,
                                  student=student,
                                  teacher=teacher,
                                  optimizer=optimizer,
                                  fp16_scaler=fp16_scaler,
                                  desd_loss=desd_loss)
    start_epoch = to_restore["epoch"]
    writer = SummaryWriter(output_dir/'tensorboard_logs.log')
    start_time = time.time()

    best_rankme = 0

    for epoch in range(start_epoch, cfg['epochs']):
        # ============ training one epoch of DeSd ... ============
        train_stats = train_one_epoch(student, teacher, desd_loss, train_data_loader,
                                      optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, cfg, writer)
        # ============ logging of training stats ... ============
        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        with (output_dir/'log.txt').open("a") as f:
            f.write(json.dumps(train_log_stats) + "\n")
        # tensorboard logging
        for k, v in train_stats.items():
            writer.add_scalar(f'train_{k}', v, epoch)

        # ============ Get metrics at the end of the epoch ... ============
        best_rankme = at_epochs_end(student, teacher, desd_loss, train_data_loader, val_data_loader,
                      epoch, fp16_scaler, cfg['metrics_cfg'], writer, best_rankme)

        # ============ writing models ... ============
        save_dict = {'student': student.state_dict(),
                     'teacher': teacher.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1,
                     'args': cfg,
                     'desd_loss': desd_loss.state_dict()}
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        torch.save(save_dict, str(output_dir/'checkpoint.pth'))
        # if cfg['saveckp_freq'] and epoch % cfg['saveckp_freq'] == 0:
        #     torch.save(save_dict, str(output_dir/f'checkpoint{epoch:04}.pth'))

        # ============ Get training time per epoch ... ============
        total_time = time.time() - start_time
        start_time = time.time()
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    # ============ Adapt checkpoint to be used as enconder in nnUNet ... ============
    adapt_chckpt(str(output_dir/'checkpoint.pth'), str(output_dir/'ssl_checkpoint.pth'))


def train_one_epoch(
    student, teacher, desd_loss, data_loader, optimizer, lr_schedule,
    wd_schedule, momentum_schedule, epoch, fp16_scaler, cfg, writer
):
    now_it = 0
    average_loss = {"total_loss": [], "loss_ssl": []}
    for i in range(student.n_heads - 1):
        average_loss[f'deep_loss_{i+1}'] = []
    data_loader.rng = np.random.default_rng(420+epoch)
    for images in data_loader:
        images = images[0]
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + now_it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        if epoch == 0:
            log_example_images(images, writer, epoch)

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            teacher_output, _, _ = teacher(images[:2])
            student_output, _, _ = student(images)
            losses = desd_loss(student_output, teacher_output, epoch)
            average_loss["total_loss"].append(losses[0].item())
            average_loss["loss_ssl"].append(losses[1].item())
            for i in range(student.n_heads - 1):
                average_loss[f'deep_loss_{i+1}'].append(losses[i+2].item())

        if not math.isfinite(losses[0].item()):
            print(f"Loss is {losses[0].item()}, stopping training")
            for i, l in enumerate(losses):
                print(f'loss {i}: {l}' )
            sys.exit(1)

        # student update
        optimizer.zero_grad()

        if fp16_scaler is None:
            losses[0].backward()
            if cfg['clip_grad']:
                _ = utils.clip_gradients(student, cfg['clip_grad'])
            utils.cancel_gradients_last_layer(epoch, student, cfg['freeze_last_layer'])
            optimizer.step()
        else:
            fp16_scaler.scale(losses[0]).backward()
            # unscale the gradients of optimizer's assigned params in-place
            if cfg['clip_grad']:
                fp16_scaler.unscale_(optimizer)
                _ = utils.clip_gradients(student, cfg['clip_grad'])
            utils.cancel_gradients_last_layer(epoch, student, cfg['freeze_last_layer'])
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                    student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if (now_it != 0) and (now_it % 100 == 0):
            text = f'TRN - [{epoch}/{cfg["epochs"]}]\t[{now_it}/{len(data_loader)}]\t'
            text = f'{text} total_loss:{round(losses[0].item(), 4)}\t' \
                f'loss_ssl:{round(losses[1].item(), 4)}\t loss_self:'
            for i in range(student.n_heads - 1):
                text = f'{text} {round(losses[i+2].item(), 4)}\t'
            text = f'{text} tlr:{round(optimizer.param_groups[0]["lr"], 6)}\t' \
                f'w:{round(optimizer.param_groups[0]["weight_decay"],4)}\n'
            print(text)
        now_it = now_it + 1

    # gather the stats from all processes
    mean_losses_epoch = {k: np.mean(meter) for k, meter in average_loss.items()}
    rounded = {k: round(v, 4) for k, v in mean_losses_epoch.items()}
    print(f"Epoch: {epoch}  -  Averaged train stats: {rounded}")
    return mean_losses_epoch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', dest='cfg_path', help='Path to the config file')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as yfile:
        cfg = yaml.safe_load(yfile)

    nnunet_cfg = {
        'num_epochs': cfg['training']['num_epochs'],
        'unfreeze_lr': cfg['training']['unfreeze_lr'],
        'unfreeze_epoch': cfg['training']['unfreeze_epoch'],
        'ssl_pretrained': False
    }
    nnunet_cfg_path = Path(nnUNet_preprocessed) / cfg['training']['dataset'] / 'nnunet_cfg.yml'
    with open(nnunet_cfg_path, 'w') as yfile:
        yaml.dump(nnunet_cfg, yfile)

    output_dir = Path(cfg['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_DeSD(cfg['training'])

    nnunet_cfg['ssl_pretrained'] = cfg['training']['ssl_pretrained']
    nnunet_cfg_path = Path(nnUNet_preprocessed) / cfg['training']['dataset'] / 'nnunet_cfg.yml'
    with open(nnunet_cfg_path, 'w') as yfile:
        yaml.dump(nnunet_cfg, yfile)
