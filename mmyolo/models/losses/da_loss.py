import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                 target=targets,
                                                 reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss