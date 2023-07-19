import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn



def dice_coef(pred, gt):
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * gt, dim=[1, 2])
    union = torch.sum(pred + gt, dim=[1, 2])
    dice_coef = (2.0 * intersection + 1e-7) / (union + 1e-7) # add a small epsilon to avoid division by 0
    dice_coef = dice_coef.mean()
    return dice_coef

def dice_loss(pred, gt):
    return 1 - dice_coef(pred, gt)


def cross_entropy_loss(pred, gt):

    loss = nn.BCEWithLogitsLoss()
    ce = loss(pred, gt)
    
    return ce


def supervised_loss(pred, gt, mu=0.5):

    d_loss = dice_loss(pred[:, 0, :, :], gt)
    ce_loss = cross_entropy_loss(pred[:, 0, :, :], gt)
    sup_loss = d_loss + mu * ce_loss
    
    return sup_loss