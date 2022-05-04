import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor, normal
from torch.jit.annotations import List

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpCosSim2d


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, sharpened_cosine_similarity, activation, normalization, memory_efficient=False,
        ):
        super(_DenseLayer, self).__init__()
        self.sharpened_cosine_similarity = sharpened_cosine_similarity
        self.activation = activation
        self.normalization = normalization
        if self.normalization:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        if self.activation:
            self.add_module('relu1', nn.ReLU(inplace=True))
        if self.sharpened_cosine_similarity:
            self.add_module('conv1', SharpCosSim2d(num_input_features,
                                        out_channels=bn_size * growth_rate,
                                        kernel_size=1,
                                        stride = 1))
        else:
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1,
                                            bias=False))
        if self.normalization:
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        if self.activation:
            self.add_module('relu2', nn.ReLU(inplace=True))
        if self.sharpened_cosine_similarity:
            self.add_module('conv2', SharpCosSim2d(bn_size * growth_rate,
                                        out_channels=growth_rate,
                                        kernel_size=1,
                                        stride = 1))
        else:
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)

        #not sure what to do here
        if self.activation and self.normalization:
            bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        elif self.activation:
            bottleneck_output = self.conv1(self.norm1(concated_features))
        elif self.normalization:
            bottleneck_output = self.conv1(self.relu1(concated_features))
        else:
            bottleneck_output = self.conv1(concated_features)

        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        if self.activation and self.normalization:
            new_features = self.conv2(self.relu2(self.norm1(bottleneck_output)))  # noqa: T484
        elif self.activation:
            new_features = self.conv2(self.norm2(bottleneck_output))
        elif self.normalization:
            new_featurest = self.conv2(self.relu2(bottleneck_output))
        else:
            new_features = self.conv2(bottleneck_output)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, sharpened_cosine_similarity, activation, normalization, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                sharpened_cosine_similarity=sharpened_cosine_similarity, 
                activation = activation, 
                normalization = normalization
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, sharpened_cosine_similarity, activation, normalization):
        super(_Transition, self).__init__()
        if normalization:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
        if activation:
            self.add_module('relu', nn.ReLU(inplace=True))
        if sharpened_cosine_similarity:
            self.add_module('conv', SharpCosSim2d(num_input_features,
                                        num_output_features,
                                        kernel_size=1,
                                        stride = 1))
        else:
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, sharpened_cosine_similarity = False, activation = True, normalization = True, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        #kinda lazy way of doing this
        #have to rewrite this lol
        f = []

        if sharpened_cosine_similarity:
            f.append(('conv0', SharpCosSim2d(3, num_init_features, kernel_size=7, stride=2, padding=3)))
        else:
            f.append(('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)))
        
        if normalization:
            f.append(('norm0', nn.BatchNorm2d(num_init_features)))
        
        if activation:
            f.append(('relu0', nn.ReLU(inplace=True)))
        

        f.append(('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
    
        self.features = nn.Sequential(OrderedDict(f))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                sharpened_cosine_similarity = sharpened_cosine_similarity,
                activation = activation,
                normalization = normalization
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, sharpened_cosine_similarity = sharpened_cosine_similarity, activation = activation, normalization = normalization)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        if normalization:
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out