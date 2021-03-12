import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_1x1_bn(
        input_channel,
        output_channel,
        bn_momentum=0.1,
        activation="relu"):
    if activation == "relu":
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel, momentum=bn_momentum),
            nn.ReLU6(inplace=True)
        )
    elif activation == "hswish":
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel, momentum=bn_momentum),
            HSwish()
        )


class Flatten(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class ConvBNRelu(nn.Sequential):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad,
                 activation="relu",
                 bn=True,
                 group=1,
                 bn_momentum=0.1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module(
            "conv",
            nn.Conv2d(
                input_channel,
                output_channel,
                kernel,
                stride,
                pad,
                groups=group,
                bias=False))
        if bn:
            self.add_module(
                "bn",
                nn.BatchNorm2d(
                    output_channel,
                    momentum=bn_momentum))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))
        elif activation == "hswish":
            self.add_module("hswish", HSwish())


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SEModule(nn.Module):
    def __init__(self, in_channel,
                 reduction=4,
                 squeeze_act=nn.ReLU(inplace=True),
                 excite_act=HSigmoid(inplace=True)):
        super(SEModule, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(in_channels=in_channel,
                                      out_channels=in_channel // reduction,
                                      kernel_size=1,
                                      bias=True)
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(in_channels=in_channel // reduction,
                                     out_channels=in_channel,
                                     kernel_size=1,
                                     bias=True)
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act


class MixedBlock(nn.Module):
    """
    Mixed Convolution block
    """
    def __init__(self,
                 input_channel,
                 output_channel,
                 stride,
                 split_block=4,
                 kernels=[3, 5, 7, 9],
                 axis=1,
                 group=1,
                 activation="relu"):
        super(MixedBlock, self).__init__()
        self.blocks = nn.ModuleList()

        self.skip_index = None
        for i, k in enumerate(kernels):
            if k == "skip":
                operation = nn.Conv2d(
                    input_channel, output_channel, 1, stride, 0, bias=False)
                self.skip_index = i
            else:
                operation = ConvBNRelu(
                    input_channel,
                    output_channel,
                    kernel=k,
                    stride=stride,
                    pad=(k // 2),
                    activation=activation,
                    group=group,
                )
            self.blocks.append(operation)
        self.split_block = split_block

        # Order without skip operation
        self.split_block = split_block - 1
        self.order = np.random.permutation(self.split_block)
        self.index = 0
        random.shuffle(self.order)
        # =============================
        self.skip_operation = False

        self.arch_param = None

    def forward(self, x, arch_flag=False):
        if not arch_flag:
            index = self.order[self.index] if not self.skip_operation else self.skip_index
            x = self.blocks[index](
                x) if index != self.skip_index else self.blocks[index](x) * 0
            return x, self.skip_operation
        else:
            block_probability = self.arch_param.cuda()
            # If skip connection, then output set 0
            x = sum(b(x) * p if i != self.skip_index else b(x) * 0 for i, (b, p)
                    in enumerate(zip(self.blocks, block_probability)) if p > 1e-2)
            return x, False

    def set_arch_param(self, arch_param):
        self.arch_param = arch_param

    def set_training_order(self, reset=False, skip=False):
        """
        Choose the convolution operation. If skip is true, choose skip operation
        """
        self.skip_operation = False
        if reset:
            self.order = np.random.permutation(self.split_block)
            random.shuffle(self.order)
            self.index = 0

        if skip:
            self.skip_operation = True
        else:
            self.index += 1
            if self.index == self.split_block:
                self.order = np.random.permutation(self.split_block)
                random.shuffle(self.order)
                self.index = 0


class MPDBlock(nn.Module):
    """Mixed path depthwise block"""

    def __init__(self,
                 input_channel,
                 output_channel,
                 stride,
                 split_block=4,
                 kernels=[3, 5, 7, 9],
                 axis=1,
                 activation="relu",
                 search=False,
                 bn_momentum=0.1):
        super(MPDBlock, self).__init__()
        self.block_input_channel = input_channel // split_block
        self.block_output_channel = output_channel // split_block

        self.split_block = len(kernels)
        self.blocks = nn.ModuleList()

        for b in range(split_block):
            if search:
                operation = MixedBlock(
                    self.block_input_channel,
                    self.block_output_channel,
                    stride=stride,
                    split_block=len(kernels),
                    kernels=kernels,
                    group=self.block_output_channel,
                    activation=activation,
                )
            else:
                operation = nn.Conv2d(
                    self.block_input_channel,
                    self.block_output_channel,
                    kernels[b],
                    stride,
                    (kernels[b] // 2),
                    groups=self.block_output_channel,
                    bias=False)
            self.blocks.append(operation)

        if not search:
            self.bn = nn.BatchNorm2d(output_channel, momentum=bn_momentum)
            if activation == "relu":
                self.activation = nn.ReLU6(inplace=True)
            elif activation == "hswish":
                self.activation = HSwish()

        self.axis = axis
        self.search = search

    def forward(self, x, arch_flag=False):
        split_x = torch.split(x, self.block_input_channel, dim=self.axis)
        # =================== SBN
        skip_connection_num = 0
        output_list = []
        skip_flag = False
        for x_i, conv_i in zip(split_x, self.blocks):
            if self.search:
                output, skip_flag = conv_i(x_i, arch_flag)
                skip_connection_num += 1 if skip_flag else 0
            else:
                output = conv_i(x_i)

            output_list.append(output)

        x = torch.cat(output_list, dim=self.axis)

        if not self.search:
            x = self.bn(x)
            x = self.activation(x)
        return x, skip_connection_num
        # ===================

    def set_arch_param(self, arch_param):
        for i, l in enumerate(self.blocks):
            l.set_arch_param(
                arch_param[i * self.split_block:(i + 1) * self.split_block])

    def set_training_order(self, active_block, reset=False, static=False):
        # ================ Choose active_block
        if static and active_block == 0:
            # At least choose one block in static layer
            active_block = random.randint(1, len(self.blocks))

        # ================ Architecture Specific
        active_order = np.random.permutation(len(self.blocks))
        active_blocks = active_order[:active_block]

        for b_num, b in enumerate(self.blocks):
            if b_num in active_blocks:
                b.set_training_order(reset, skip=False)
            else:
                b.set_training_order(reset, skip=True)


class MBConv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 expansion,
                 kernels,
                 stride,
                 activation,
                 min_expansion=2,
                 split_block=1,
                 group=1,
                 se=False,
                 search=False,
                 bn_momentum=0.1,
                 *args,
                 **kwargs):
        super(MBConv, self).__init__()
        self.use_res_connect = True if (
            stride == 1 and input_channel == output_channel) else False
        mid_depth = int(input_channel * expansion)

        self.group = group

        if input_channel == mid_depth:
            self.point_wise = nn.Sequential()
        else:
            self.point_wise = ConvBNRelu(input_channel,
                                         mid_depth,
                                         kernel=1,
                                         stride=1,
                                         pad=0,
                                         activation=activation,
                                         group=group,
                                         bn_momentum=bn_momentum
                                         )

        self.depthwise = MPDBlock(mid_depth,
                                  mid_depth,
                                  stride,
                                  split_block=expansion,
                                  kernels=kernels,
                                  activation=activation,
                                  search=search,
                                  bn_momentum=bn_momentum)

        self.point_wise_1 = ConvBNRelu(mid_depth,
                                       output_channel,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       bn=False if search else True,
                                       group=group,
                                       bn_momentum=bn_momentum
                                       )
        self.se = SEModule(mid_depth) if se else None

        self.expansion = expansion
        self.search = search
        if self.search:
            # ====================
            # reference : https://arxiv.org/pdf/2001.05887.pdf
            self.sbn = nn.ModuleList()
            for i in range(expansion):
                self.sbn.append(nn.BatchNorm2d(output_channel))
            self.skip_connection_num = 0
            # ===================

            # =================== Training order
            self.index = 0

            self.order = [i for i in range(min_expansion, expansion+1)]
            self.order = np.array(self.order)
            random.shuffle(self.order)
            # ===================

    def forward(self, x, arch_flag=False):
        y = self.point_wise(x)
        y, skip_connection_num = self.depthwise(
            y) if not arch_flag else self.depthwise(y, arch_flag)

        y = self.se(y) if self.se is not None else y
        y = self.point_wise_1(y)

        if self.search:
            # ============== SBN
            if arch_flag:
                skip_connection_num = round(self.skip_connection_num)
            if skip_connection_num != self.expansion:
                y = self.sbn[skip_connection_num](y)
            # ==============

        y = y + x if self.use_res_connect else y

        return y

    def set_arch_param(self, arch_param):
        self.depthwise.set_arch_param(arch_param)

        # Count continue skip num for updating architecture generator
        self.skip_connection_num = 0
        kernel_nums = len(arch_param) // self.expansion
        for i in range(0, len(arch_param), kernel_nums):
            split_arch_param = arch_param[i:i + kernel_nums]
            self.skip_connection_num += split_arch_param[-1].item()

    def set_training_order(self, reset=False, state=None, static=False):
        expansion = None
        if reset:
            self.index = 0
            random.shuffle(self.order)

            expansion = self.order[self.index]
            self.index += 1
        else:
            expansion = self.order[self.index]

            self.index += 1
            if self.index == len(self.order):
                self.index = 0
                random.shuffle(self.order)

        self.depthwise.set_training_order(expansion, reset, static)


if __name__ == "__main__":
    i = torch.zeros((1, 3, 32, 32))
    block = MBConv(3, 16, 6, [3], 2, "relu", split_block=1)
