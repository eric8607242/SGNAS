import os
import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient
from utils.prior_pool import PriorPool
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import get_optimizer, get_lr_scheduler, CrossEntropyLossSoft
from utils.lookup_table_builder import LookUpTable
from utils.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="path to the config file",
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

    get_logger(CONFIG.log_dir)
    writer = get_writer(args.title, CONFIG.write_dir)

    logging.info(
        "=================================== Experiment title : {} Start ===========================".format(
            args.title))

    set_random_seed(CONFIG.seed)

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(
        train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, CONFIG)

    model = Supernet(CONFIG)
    lookup_table = LookUpTable(CONFIG)

    criterion = cross_encropy_with_label_smoothing

    arch_param_nums = model.get_arch_param_nums()
    generator = get_generator(CONFIG, arch_param_nums)

    if CONFIG.model_pretrained is not None and os.path.isfile(
            CONFIG.model_pretrained):
        logging.info(
            "Load pretrained weight from {}".format(
                CONFIG.model_pretrained))
        model.load_state_dict(torch.load(CONFIG.model_pretrained)["model"])

    model.to(device)
    generator.to(device)
    if (device.type == "cuda" and CONFIG.ngpu >= 1):
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

    prior_pool = PriorPool(
        lookup_table,
        arch_param_nums,
        generator,
        model,
        test_loader,
        CONFIG)

    optimizer = get_optimizer(model, CONFIG.optim_state)
    g_optimizer = get_optimizer(generator, CONFIG.g_optim_state)
    scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

    start_time = time.time()
    trainer = Trainer(
        criterion,
        optimizer,
        g_optimizer,
        scheduler,
        writer,
        device,
        lookup_table,
        prior_pool,
        CONFIG)
    trainer.search_train_loop(
        train_loader,
        val_loader,
        val_loader,
        model,
        generator)
    logging.info("Total search time: {:.2f}".format(time.time() - start_time))

    logging.info(
        "=================================== Experiment title : {} End ===========================".format(
            args.title))
