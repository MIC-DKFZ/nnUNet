############
# https://github.com/lessw2020/mish/blob/master/mish.py
# This code was taken from the repo above and was not created by me (Fabian)! Full credit goes to the original authors
############

import torch

import torch.nn as nn
import torch.nn.functional as F  # (uncomment if needed,but you likely already have it)


# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020
# github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))
