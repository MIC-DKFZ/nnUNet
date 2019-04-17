#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import torch


def hard_dice(output, target):
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    target = target[:, 0]
    # target is not one hot encoded, output is
    # target must be the CPU segemtnation, not tensor. output is pytorch tensor
    num_classes = output.shape[1]
    output = output.argmax(1)
    foreground_classes = np.arange(1, num_classes)
    all_tp = []
    all_fp = []
    all_fn = []
    all_fg_dc = []
    for s in range(target.shape[0]):
        tp = []
        fp = []
        fn = []
        for c in foreground_classes:
            t_is_c = target[s] == c
            o_is_c = output[s] == c
            t_is_not_c = target[s] != c
            o_is_not_c = output[s] != c
            tp.append(np.sum(o_is_c & t_is_c))
            fp.append(np.sum(o_is_c & t_is_not_c))
            fn.append(np.sum(o_is_not_c & t_is_c))
        foreground_dice = [2 * i / (2 * i + j + k + 1e-8) for i, j, k in zip(tp, fp, fn)]
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_fg_dc.append(foreground_dice)
    return all_fg_dc, all_tp, all_fp, all_fn
