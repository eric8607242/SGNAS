import logging

import torch
import torch.nn as nn


def get_lr_scheduler(optimizer, step_per_epoch, CONFIG):
    logging.info("================ Scheduler =================")
    logging.info("Scheduler : {}".format(CONFIG.lr_scheduler))
    if CONFIG.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=step_per_epoch * CONFIG.epochs)
    elif CONFIG.lr_scheduler == "step":
        logging.info("Step size : {}".format(CONFIG.step_size))
        logging.info("Gamma : {}".format(CONFIG.decay_ratio))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=CONFIG.step_size, gamma=CONFIG.decay_ratio, last_epoch=-1)

    return lr_scheduler


def get_optimizer(model, CONFIG, log_info=""):
    logging.info("================= Optimizer =================")
    logging.info("{} Optimizer : {}".format(log_info, CONFIG.optimizer))
    logging.info("{} Learning rate : {}".format(log_info, CONFIG.lr))
    logging.info("{} Weight decay : {}".format(log_info, CONFIG.weight_decay))
    if CONFIG.optimizer == "sgd":
        logging.info("{} Momentum : {}".format(log_info, CONFIG.momentum))
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=CONFIG.lr,
                                    momentum=CONFIG.momentum,
                                    weight_decay=CONFIG.weight_decay)

    elif CONFIG.optimizer == "rmsprop":
        logging.info("{} Momentum : {}".format(log_info, CONFIG.momentum))
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=CONFIG.lr,
                                        alpha=CONFIG.alpha,
                                        momentum=CONFIG.momentum,
                                        weight_decay=CONFIG.weight_decay)
    elif CONFIG.optimizer == "adam":
        logging.info("{} Beta : {}".format(log_info, CONFIG.beta))
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=CONFIG.weight_decay,
                                     lr=CONFIG.lr,
                                     betas=(CONFIG.beta, 0.999))

    return optimizer


def cal_hc_loss(generate_hc, target_hc, alpha, loss_penalty):
    if generate_hc > target_hc + 0.1:
        return (generate_hc - target_hc)**2 * alpha * loss_penalty
    elif generate_hc < target_hc - 0.1:
        return (target_hc - generate_hc)**2 * alpha
    else:
        return (target_hc - generate_hc)**2 * 0


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return torch.mean(cross_entropy_loss)
