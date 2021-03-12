import time
import random
import logging
import argparse

import torch
import torch.nn as nn

import pandas as pd
import scipy.stats as stats

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient, min_max_normalize
from utils.model import Model
from utils.prior_pool import PriorPool
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import cal_hc_loss, get_optimizer, get_lr_scheduler
from utils.lookup_table_builder import LookUpTable


def evaluate_generator(
        generator,
        prior_pool,
        lookup_table,
        CONFIG,
        device,
        val=True):
    """
    Evaluate kendetall and hardware constraint loss of generator
    """
    total_loss = 0

    evaluate_metric = {"gen_flops": [], "true_flops": []}
    for mac in range(CONFIG.low_flops, CONFIG.high_flops, 10):
        hardware_constraint = torch.tensor(mac, dtype=torch.float32)
        hardware_constraint = hardware_constraint.view(-1, 1)
        hardware_constraint = hardware_constraint.to(device)

        prior = prior_pool.get_prior(hardware_constraint.item())
        prior = prior.to(device)

        normalize_hardware_constraint = min_max_normalize(
            CONFIG.high_flops, CONFIG.low_flops, hardware_constraint)

        arch_param = generator(prior, normalize_hardware_constraint)
        arch_param = lookup_table.get_validation_arch_param(arch_param)

        layers_config = lookup_table.decode_arch_param(arch_param)

        gen_mac = lookup_table.get_model_flops(arch_param)
        hc_loss = cal_hc_loss(
            gen_mac.cuda(),
            hardware_constraint.item(),
            CONFIG.alpha,
            CONFIG.loss_penalty)

        evaluate_metric["gen_flops"].append(gen_mac.item())
        evaluate_metric["true_flops"].append(mac)

        total_loss += hc_loss.item()
    tau, _ = stats.kendalltau(
        evaluate_metric["gen_flops"], evaluate_metric["true_flops"])

    return evaluate_metric, total_loss, tau


def evaluate_lookup_table(lookup_table, prior_pool, CONFIG, evaluate_nums=10):
    for i in range(evaluate_nums):
        gen_mac, arch_param = prior_pool.generate_arch_param(lookup_table)
        gen_mac = lookup_table.get_model_flops(arch_param.cuda())
        layers_config = lookup_table.decode_arch_param(arch_param)

        model = Model(layers_config, CONFIG.dataset, CONFIG.classes)

        cal_model_efficient(model, CONFIG)


def save_generator_evaluate_metric(
        evaluate_metric,
        path_to_generator_evaluate):
    df_metric = pd.DataFrame(evaluate_metric)
    df_metric.to_csv(path_to_generator_evaluate, index=False)
