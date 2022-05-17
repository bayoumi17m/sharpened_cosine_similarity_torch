'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from sharpened_cosine_similarity import SharpenedCosineSimilarity
from absolute_pooling import MaxAbsPool2d

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and init_weights:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGNoDropoutConv(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, init_weights=True):
        super(VGGNoDropoutConv, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and init_weights:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_scs_layers(cfg, batch_norm=False, use_relu=True, abspool=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            if abspool:
                layers += [MaxAbsPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = SharpenedCosineSimilarity(
                in_channels=in_channels,
                out_channels=v,
                kernel_size=3,
                padding=1
            )
            layers += [conv2d]

            if batch_norm:
                layers += [nn.BatchNorm2d(v)]

            if use_relu:
                layers += [nn.ReLU(inplace=True)]

            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

def vgg():
    """
    VGG 11-layer model (configuration "A") with:
    - Standard conv
    - dropout
    - relu activation
    - batch norm
    """
    return VGG(make_layers(cfg['A'], batch_norm=True, use_relu=True), init_weights=True)

def vgg_scs_bn_act_do():
    """
    - SCS conv
    - dropout
    - batch norm
    - relu activation
    """
    return VGG(make_scs_layers(cfg['A'], batch_norm=True, use_relu=True), init_weights=False)

def vgg_scs_bn_do():
    """
    - SCS conv
    - dropout
    - batch norm
    """
    return VGG(make_scs_layers(cfg['A'], batch_norm=True, use_relu=False), init_weights=False)


def vgg_scs_bn_act():
    """
    - SCS conv
    - batch norm
    - relu activation
    """
    return VGGNoDropoutConv(make_scs_layers(cfg['A'], batch_norm=True, use_relu=True), init_weights=False)

def vgg_scs_bn_act_do_abspool():
    """
    - SCS conv
    - dropout
    - batch norm
    - relu activation
    - abs pool
    """
    return VGG(make_scs_layers(cfg['A'], batch_norm=True, use_relu=True, abspool=True), init_weights=False)

def vgg_scs_bn_abspool():
    """
    - SCS conv
    - batch norm
    - abs pooling
    """
    return VGGNoDropoutConv(make_scs_layers(cfg['A'], batch_norm=True, use_relu=False, abspool=True), init_weights=False)

def vgg_scs_bn_act_abspool():
    """
    - SCS conv
    - batch norm
    - abs pooling
    """
    return VGGNoDropoutConv(make_scs_layers(cfg['A'], batch_norm=True, use_relu=True, abspool=True), init_weights=False)

def vgg_scs_bn():
    """
    - SCS conv
    - batch norm
    - max pooling
    """
    return VGGNoDropoutConv(make_scs_layers(cfg['A'], batch_norm=True, use_relu=False, abspool=False), init_weights=False)