import json
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import MBConv, ConvBNRelu, conv_1x1_bn


class Supernet(nn.Module):
    def __init__(self, CONFIG):
        super(Supernet, self).__init__()
        self.CONFIG = CONFIG
        self.classes = CONFIG.classes
        self.dataset = CONFIG.dataset

        if self.dataset[:5] == "cifar":
            first_stride = 1
        elif self.dataset == "imagenet" or self.dataset == "imagenet_lmdb":
            first_stride = 2

        self.first = ConvBNRelu(
            input_channel=3,
            output_channel=32,
            kernel=3,
            stride=first_stride,
            pad=3 // 2,
            activation="relu")

        input_channel = 32
        output_channel = 16
        self.first_mb = MBConv(input_channel=input_channel,
                               output_channel=output_channel,
                               expansion=1,
                               kernels=[3],
                               stride=1,
                               activation="relu",
                               min_expansion=1,
                               split_block=1,
                               se=False)

        input_channel = output_channel
        self.stages = nn.ModuleList()

        for l_cfg in self.CONFIG.l_cfgs:
            min_expansion = self.CONFIG.min_expansion
            expansion, output_channel, kernels, stride, split_block, se = l_cfg
            self.stages.append(MBConv(input_channel=input_channel,
                                      output_channel=output_channel,
                                      expansion=expansion,
                                      kernels=kernels,
                                      stride=stride,
                                      activation="relu",
                                      min_expansion=min_expansion,
                                      split_block=expansion,
                                      se=se,
                                      search=True))
            input_channel = output_channel

        self.last_stage = conv_1x1_bn(input_channel, 1280)
        self.classifier = nn.Linear(1280, self.classes)

        self.split_block = split_block
        self.architecture_param_nums = len(
            self.CONFIG.l_cfgs) * (self.CONFIG.split_blocks * self.CONFIG.kernels_nums)

        self._initialize_weights()

    def forward(self, x, arch_flag=False):
        x = self.first(x)
        x = self.first_mb(x)
        for i, l in enumerate(self.stages):
            x = l(x) if not arch_flag else l(x, arch_flag)
        x = self.last_stage(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

    def get_arch_param_nums(self):
        return self.architecture_param_nums

    def set_arch_param(self, arch_param):
        for i, l in enumerate(self.stages):
            l.set_arch_param(arch_param[i])
        return arch_param

    def set_training_order(self, reset=False, state=None):
        for l_num, l in enumerate(self.stages):
            if l_num in self.CONFIG.static_layers:
                l.set_training_order(reset, state, static=True)
            else:
                l.set_training_order(reset, state)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
