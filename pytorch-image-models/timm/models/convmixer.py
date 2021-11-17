import sys
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from DCLS.construct.modules.Dcls import  Dcls2d as cDcls2d
from DCLS.modules.Dcls import  Dcls2d
import math

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

def ConvMixerDcls(dim, depth, kernel_size=9, dilation=1, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    cDcls2d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding=dilation * (kernel_size - 1) // 2, gain=0.9/(math.sqrt(dim)*kernel_size)),
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
@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16_5x5(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=5,  patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16_7x7(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=7,  patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=9,  patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=9, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_1536_20_dilated(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=3, dilation=3, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


@register_model
def convmixer_768_32_dilated(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=3, dilation=3, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16_dilated(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=3, dilation=3, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8_dilated(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=3, dilation=3, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_1536_20_dcls(pretrained=False, **kwargs):
    model = ConvMixerDcls(1536, 20, kernel_size=3, dilation=3, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_768_32_dcls(pretrained=False, **kwargs):
    model = ConvMixerDcls(768, 32, kernel_size=3, dilation=3, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16_dcls(pretrained=False, **kwargs):
    model = ConvMixerDcls(256, 16, kernel_size=3, dilation=3, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8_dcls(pretrained=False, **kwargs):
    model = ConvMixerDcls(256, 8, kernel_size=3, dilation=3, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model
