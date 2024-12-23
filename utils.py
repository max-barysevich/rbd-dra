# -*- coding: utf-8 -*-

import torch
from torch import nn

#%%

class Charbonnier(nn.Module):
    
    def __init__(self,epsilon=1e-4):
        super().__init__()
        self.eps = epsilon
    
    def forward(self,y_pred,y_true):
        diff = y_true - y_pred
        return ((diff**2 + self.eps**2)**.5).mean()

def psnr(gt,output):
    mse = torch.mean((gt-output)**2)
    return 10*torch.log10((gt.max()**2) / mse)