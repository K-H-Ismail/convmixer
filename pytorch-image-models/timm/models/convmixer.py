import sys

import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from DCLS.construct.modules.Dcls import  Dcls2d as cDcls2d
import math
import torch


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, dilation=1, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding=dilation * (kernel_size - 1) // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


def ConvMixerDcls(dim, depth, kernel_count=3, dilated_kernel_size=9, scaling=1, dcls_sync=False, groups=256, patch_size=7, n_classes=1000):
    model = nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    cDcls2d(dim, dim, kernel_count=kernel_count, dilated_kernel_size=dilated_kernel_size, 
                            groups=groups, padding=(dilated_kernel_size - 1) // 2, scaling=scaling),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

    if dcls_sync:
        P = torch.Tensor(2, dim, dim // groups, kernel_count) 
        with torch.no_grad():
            lim = (dilated_kernel_size//2)
            scaling = 1
            torch.nn.init.normal_(P, 0, 0.5).clamp_(-lim,lim).div_(scaling)
            #P = P.repeat(1, dim, dim // groups, 1)
            P = torch.nn.parameter.Parameter(P.detach().clone())

        for i in range(depth):        
            model[i+3][0].fn[0].P = P
    return model

@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=9,  patch_size=1, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg    
    return model

@register_model
def convmixer_256_16_dcls(pretrained=False, **kwargs):
    kernel_count = kwargs.get('dcls_kernel_count', 11)
    dilated_kernel_size = kwargs.get('dcls_kernel_size', 11)
    dcls_sync = kwargs.get('dcls_sync', False)
    scaling = 1
    dim = 256
    groups = dim
    depth = 16
    model = ConvMixerDcls(dim, depth, kernel_count=kernel_count,  dilated_kernel_size=dilated_kernel_size, scaling=scaling, dcls_sync=dcls_sync,  groups = groups,  patch_size=1, n_classes=kwargs.get("num_classes"))
    model.default_cfg = _cfg    
    return model
