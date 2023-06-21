import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class DeSDLoss(nn.Module):
    def __init__(self, out_dim, ncrops, n_heads, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, weights=None):
        super().__init__()
        self.student_temp = student_temp
        self.n_heads = n_heads
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("loss_center", torch.zeros(1, 4))
        self.weights = weights

        if self.weights == 'constant':
            self.weights = np.ones(n_heads)
        elif self.weights == 'exp':
            self.weights = np.arange(n_heads)
            self.weights = np.exp((-0.5)*self.weights)
        elif self.weights == 'fraction':
            self.weights = np.arange(n_heads)+0.25
            self.weights = 1 / self.weights
        elif self.weights == 'linear':
            self.weights = np.arange(n_heads) / (n_heads+1)
            self.weights = 1 - self.weights
        elif self.weights == 'last':
            self.weights = np.array([1, 0, 0, 0, 0, 0])
        else:
            self.weights = np.array([0.25, 0.25, 0.25, 0.25, 0, 0])
        self.weights = (self.weights / np.sum(self.weights))[::-1]

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        for i in range(self.n_heads):
            student_output[i] = (student_output[i] / self.student_temp).chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out_self = F.softmax(
            (teacher_output[-1] - self.center) / temp, dim=-1).detach().chunk(2)
        loss = [0] * len(student_output)
        n_loss_terms = 0

        for iq in range(len(teacher_out_self)):
            for v in range(len(student_output[-1])):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                for i in range(self.n_heads):
                    st_o = F.log_softmax(student_output[i][v], dim=-1)
                    loss[i] += torch.sum(
                        -teacher_out_self[iq] * st_o,
                        dim=-1).mean()
                n_loss_terms += 1
        
        self.update_center(teacher_output[-1])

        loss = [loss_i / n_loss_terms for loss_i in loss]

        total_loss = 0
        for i in range(len(loss)):
            total_loss += loss[i] * self.weights[i]

        return tuple([total_loss, loss[-1]] + loss[:-1])

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)

