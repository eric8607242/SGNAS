import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient, min_max_normalize
from utils.prior_pool import PriorPool
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import get_optimizer, get_lr_scheduler
from utils.lookup_table_builder import LookUpTable
from utils.trainer import Trainer
from utils.model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="path to the config file",
        required=True)
    parser.add_argument(
        "--flops",
        type=float,
        help="flops for the sample architecture",
        required=True)
    parser.add_argument(
        "--title",
        type=str,
        help="experiment title",
        required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device(
            "cuda" if (
                torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    set_random_seed(CONFIG.seed)

    get_logger(CONFIG.log_dir)
    writer = get_writer(args.title, CONFIG.write_dir)

    logging.info(
        "=================================== Experiment title : {} Start ===========================".format(
            args.title))

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(
        train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, test_dataset, test_dataset, CONFIG)

    lookup_table = LookUpTable(CONFIG)

    supernet = Supernet(CONFIG)
    arch_param_nums = supernet.get_arch_param_nums()

    generator = get_generator(CONFIG, arch_param_nums)

    criterion = cross_encropy_with_label_smoothing

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
    prior = prior_pool.get_prior(args.flops)
    prior = prior.to(device)

    noise = torch.randn(*prior.shape)
    noise = noise.to(device)
    noise *= 0

    hardware_constraint = torch.tensor(args.flops).to(device)
    normalize_hardware_constraint = min_max_normalize(
        CONFIG.high_flops, CONFIG.low_flops, hardware_constraint)

    arch_param = generator(prior, normalize_hardware_constraint, noise)
    arch_param = lookup_table.get_validation_arch_param(arch_param)

    gen_flops = lookup_table.get_model_flops(arch_param)

    logging.info("Generate flops : {}".format(gen_flops))

    layers_config = lookup_table.decode_arch_param(arch_param)
    model = Model(
        l_cfgs=layers_config,
        dataset=CONFIG.dataset,
        classes=CONFIG.classes)
    cal_model_efficient(model, CONFIG)
    if (device.type == "cuda" and CONFIG.ngpu >= 1):
        model = model.to(device)
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))
    # ============================

    optimizer = get_optimizer(model, CONFIG.optim_state)
    scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

    start_time = time.time()
    trainer = Trainer(
        criterion,
        optimizer,
        None,
        scheduler,
        writer,
        device,
        lookup_table,
        prior_pool,
        CONFIG)
    trainer.train_loop(train_loader, test_loader, model)
    logging.info(
        "Total training time : {:.2f}".format(
            time.time() -
            start_time))
    logging.info(
        "=================================== Experiment title : {} End ===========================".format(
            args.title))
