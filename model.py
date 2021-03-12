import re
import math
from copy import deepcopy

import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cal_model_efficient, min_max_normalize
from utils.prior_pool import PriorPool
from utils.lookup_table_builder import LookUpTable
from utils.model import Model


def get_model(
        config_path,
        target_flops,
        num_classes=1000,
        in_chans=3,
        activation="relu",
        se=False,
        bn_momentum=0.1):
    CONFIG = get_config(config_path)
    if CONFIG.cuda:
        device = torch.device(
            "cuda" if (
                torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    lookup_table = LookUpTable(CONFIG)

    supernet = Supernet(CONFIG)
    arch_param_nums = supernet.get_arch_param_nums()

    generator = get_generator(CONFIG, arch_param_nums)

    if CONFIG.generator_pretrained is not None:
        generator.load_state_dict(torch.load(
            CONFIG.generator_pretrained)["model"])

    generator.to(device)
    prior_pool = PriorPool(
        lookup_table,
        arch_param_nums,
        None,
        None,
        None,
        CONFIG)

    # Sample architecture parameter =======================
    prior = prior_pool.get_prior(target_flops)
    prior = prior.to(device)

    hardware_constraint = torch.tensor(target_flops).to(device)
    normalize_hardware_constraint = min_max_normalize(
        CONFIG.high_flops, CONFIG.low_flops, hardware_constraint)

    arch_param = generator(prior, normalize_hardware_constraint)
    arch_param = lookup_table.get_validation_arch_param(arch_param)

    gen_flops = lookup_table.get_model_flops(arch_param)

    logging.info("Generate flops : {}".format(gen_flops))

    layers_config = lookup_table.decode_arch_param(arch_param)
    model = Model(
        l_cfgs=layers_config,
        dataset=CONFIG.dataset,
        classes=CONFIG.classes,
        activation=activation,
        se=se,
        bn_momentum=bn_momentum)

    cal_model_efficient(model, CONFIG)
    return model
