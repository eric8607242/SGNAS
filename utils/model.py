import math

import torch
import torch.nn as nn

from utils.network_utils import ConvBNRelu, MBConv, conv_1x1_bn

BASIC_CFGS = [[3, 32, [3, 5, 7, 9], 1, 4, False],
              [3, 32, [3, 5, 7, 9], 1, 4, False],
              [3, 40, [3, 5, 7, 9], 2, 4, False],
              [3, 40, [3, 5, 7, 9], 1, 4, False],
              [3, 40, [3, 5, 7, 9], 1, 4, False],
              [3, 40, [3, 5, 7, 9], 1, 4, False],
              [3, 80, [3, 5, 7, 9], 2, 4, False],
              [3, 80, [3, 5, 7, 9], 1, 4, False],
              [3, 80, [3, 5, 7, 9], 1, 4, False],
              [3, 80, [3, 5, 7, 9], 1, 4, False],
              [3, 96, [3, 5, 7, 9], 1, 4, False],
              [3, 96, [3, 5, 7, 9], 1, 4, False],
              [3, 96, [3, 5, 7, 9], 1, 4, False],
              [3, 96, [3, 5, 7, 9], 1, 4, False],
              [3, 192, [3, 5, 7, 9], 2, 4, False],
              [3, 192, [3, 5, 7, 9], 1, 4, False],
              [3, 192, [3, 5, 7, 9], 1, 4, False],
              [3, 192, [3, 5, 7, 9], 1, 4, False],
              [3, 320, [3, 5, 7, 9], 1, 4, False]]


class Model(nn.Module):
    def __init__(
            self,
            se=False,
            activation="relu",
            bn_momentum=0.1,
            l_cfgs=BASIC_CFGS,
            dataset="imagenet",
            classes=1000):
        super(Model, self).__init__()
        if dataset[:5] == "cifar":
            self.first = ConvBNRelu(
                input_channel=3,
                output_channel=32,
                kernel=3,
                stride=1,
                pad=3 // 2,
                activation=activation,
                bn_momentum=bn_momentum)
        elif dataset[:8] == "imagenet":
            self.first = ConvBNRelu(
                input_channel=3,
                output_channel=32,
                kernel=3,
                stride=2,
                pad=3 // 2,
                activation=activation,
                bn_momentum=bn_momentum)

        input_channel = 32
        output_channel = 16
        self.first_mb = MBConv(input_channel=input_channel,
                               output_channel=output_channel,
                               expansion=1,
                               kernels=[3],
                               stride=1,
                               activation=activation,
                               split_block=1,
                               se=se,
                               bn_momentum=bn_momentum)

        input_channel = output_channel
        self.stages = nn.ModuleList()
        for l_cfg in l_cfgs:
            expansion, output_channel, kernel, stride, split_block, _ = l_cfg
            self.stages.append(MBConv(input_channel=input_channel,
                                      output_channel=output_channel,
                                      expansion=expansion,
                                      kernels=kernel,
                                      stride=stride,
                                      activation=activation,
                                      split_block=split_block,
                                      se=se,
                                      bn_momentum=bn_momentum))
            input_channel = output_channel

        self.last_stage = conv_1x1_bn(
            input_channel,
            1280,
            activation=activation,
            bn_momentum=bn_momentum)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.first(x)
        x = self.first_mb(x)
        for l in self.stages:
            x = l(x)
        x = self.last_stage(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = Model(dataset="cifar10", classes=100)
    a = torch.zeros((1, 3, 32, 32))
