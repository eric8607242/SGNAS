import sys
import os
import time
import logging
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from utils.countflops import FLOPS_Counter


class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def set_random_seed(seed):
    import random
    logging.info("Set seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_dir=None):
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_format = "%(asctime)s | %(message)s"

    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt="%m/%d %I:%M:%S %p")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "logger"))
        file_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(file_handler)


def get_writer(title, writer_dir=None):
    today = datetime.today()
    current_time = today.strftime("%d%m%Y%H%M%S")
    writer_dir = os.path.join(writer_dir, current_time + "_{}".format(title))

    writer = SummaryWriter(log_dir=writer_dir)
    return writer


def cal_model_efficient(model, CONFIG):
    latency = calculate_latency(model, 3, CONFIG.input_size)
    counter = FLOPS_Counter(
        model, [1, 3, CONFIG.input_size, CONFIG.input_size])
    flops = counter.print_summary()
    param_nums = calculate_param_nums(model)

    logging.info("INference time : {:.5f}".format(latency))
    logging.info("FLOPSs numbers(M) : {}".format(flops["total_gflops"] * 1000))
    logging.info("Parameter numbers : {}".format(param_nums))

    return flops["total_gflops"] * 1000


def calculate_latency(model, input_channel, input_size):
    input_sample = torch.randn((1, input_channel, input_size, input_size))

    start_time = time.time()
    model(input_sample)
    inference_time = time.time() - start_time

    return inference_time


def calculate_param_nums(model):
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    return total_params


def accuracy(output, target, topk=(1,)):
    """Compute the precision for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save(model, optimizer, model_path):
    torch.save({"model": model.module.state_dict() if isinstance(model,
                                                                 nn.DataParallel) else model.state_dict(),
                "optimzier": optimizer.state_dict()},
               model_path)


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def label_smoothing(pred, target, eta=0.1):
    '''
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    '''
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_encropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)


def min_max_normalize(max_value, min_value, value):
    n_value = (value - min_value) / (max_value - min_value)
    return n_value


def bn_calibration(m, cumulative_bn_stats=True):
    """Recalculate BN's running statistics.

    Should be called like "model.apply(bn_calibration)".
    Args:
        m : sub_module to dealt with.
        cumulative_bn_stats: "True" to usage arithemetic mean instead of EMA.
    """
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_running_stats()
        m.train()
        if cumulative_bn_stats:
            m.momentum = None
