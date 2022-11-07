import torch
import torch.nn.functional as F
from torch import nn

_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def forward(self, logits_4D, labels_4D):
        if len(labels_4D.shape) == 4:
            labels_4D = labels_4D[:, 0]
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        # loss = self.forward_softmax_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):
        """
        Using both softmax and sigmoid operations.
        Args:
            logits_4D     :    [N, C, H, W], dtype=float32
            labels_4D     :    [N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=_CLIP_MIN, max=_CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D     :    [N, C, H, W], dtype=float32
            labels_4D     :    [N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D     :    [N, C, H, W], dtype=float32
            probs_4D     :    [N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #    rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
    """get map pairs
    Args:
        labels_4D    :    labels, shape [N, C, H, W]
        probs_4D    :    probabilities, shape [N, C, H, W]
        radius        :    the square radius
    Return:
        tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
    """
    # pad to ensure the following slice operation is valid
    # pad_beg = int(radius // 2)
    # pad_end = radius - pad_beg

    # the original height and width
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    # https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
    # padding = (pad_beg, pad_end, pad_beg, pad_end)
    # labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

    # get the neighbors
    la_ns = []
    pr_ns = []
    # for x in range(0, radius, 1):
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        # for calculating RMI
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        # for other purpose
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors


def log_det_by_cholesky(matrix):
    """
    Args:
        matrix: matrix must be a positive define matrix.
                shape [N, C, D, D].
    Ref:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
    """
    # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
    # where C is the cholesky decomposition of A.
    # chol = torch.cholesky(matrix)
    chol = torch.linalg.cholesky(matrix)

    # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

