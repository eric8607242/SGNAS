import math

import torch
import torch.nn as nn

class ConvRelu(nn.Sequential):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad):
        super(ConvRelu, self).__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                input_channel,
                output_channel,
                kernel,
                stride,
                pad,
                bias=False))
        self.add_module("relu", nn.ReLU())

class SinGenerator(nn.Module):
    def __init__(
            self,
            hc_dim,
            input_dim,
            layer_nums=5,
            hidden_dim=32,
            min_hidden_dim=32):
        super(SinGenerator, self).__init__()
        N = hidden_dim
        self.hc_dim = hc_dim

        self.hc_head = nn.Sequential(
            ConvRelu(hc_dim, N, 3, 1, 1),
            ConvRelu(N, N, 3, 1, 1),
            ConvRelu(N, N, 3, 1, 1))
        self.bb_head = nn.Sequential(
            ConvRelu(input_dim, N, 3, 1, 1),
            ConvRelu(N, N, 3, 1, 1))

        self.head = ConvRelu(N, N, 3, 1, 1)

        self.body = nn.Sequential()
        for i in range(layer_nums - 2):
            N = int(hidden_dim / pow(2, i + 1))
            block = ConvRelu(max(2 * N, min_hidden_dim),
                             max(N, min_hidden_dim), 3, 1, 1)
            self.body.add_module("block%d" % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, min_hidden_dim), input_dim, 3, 1, 1)
        )

        self._initialize_weights()

    def forward(self, x, hc):
        y = x.view(1, 1, *x.shape)

        hc = hc.expand(*hc.shape[:-1], self.hc_dim * x.size(-1) * x.size(-2))
        hc = hc.view(1, self.hc_dim, *x.shape[-2:])
        hc_test = self.hc_head(hc)

        y = self.bb_head(y)
        prior_y = y + hc_test

        y_head = self.head(prior_y)
        y = self.body(y_head)
        y = y + prior_y
        y = self.tail(y)

        return y

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


def get_generator(CONFIG, arch_param_nums=None):
    generator = None
    if CONFIG.generator == "singan":
        generator = SinGenerator(CONFIG.hc_dim, 1)
    else:
        raise

    return generator
