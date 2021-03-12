import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn

from utils.evaluate import evaluate_generator, save_generator_evaluate_metric
from utils.util import AverageMeter, save, accuracy, min_max_normalize, bn_calibration
from utils.optim import cal_hc_loss


class Trainer:
    def __init__(
            self,
            criterion,
            optimizer,
            g_optimizer,
            scheduler,
            writer,
            device,
            lookup_table,
            prior_pool,
            CONFIG):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.hc_losses = AverageMeter()

        self.writer = writer
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.g_optimizer = g_optimizer
        self.scheduler = scheduler

        self.CONFIG = CONFIG

        self.epochs = self.CONFIG.epochs
        self.warmup_epochs = self.CONFIG.warmup_epochs
        self.search_epochs = self.CONFIG.search_epochs

        self.prior_pool = prior_pool
        # ==============
        self.hardware_pool = [
            i for i in range(
                self.CONFIG.low_flops,
                self.CONFIG.high_flops,
                5)]
        self.hardware_index = 0
        random.shuffle(self.hardware_pool)
        # ==============

        self.lookup_table = lookup_table

    def search_train_loop(
            self,
            train_loader,
            val_loader,
            test_loader,
            model,
            generator):
        self.epochs = self.warmup_epochs + self.search_epochs

        # Training supernet
        best_top1 = 0.0
        for epoch in range(self.warmup_epochs):
            logging.info(
                "Learning Rate: {:.4f}".format(
                    self.optimizer.param_groups[0]["lr"]))
            self.writer.add_scalar(
                "learning_rate/weights",
                self.optimizer.param_groups[0]["lr"],
                epoch)
            logging.info("Start to train for warmup epoch {}".format(epoch))

            self._training_step(model, train_loader, epoch,
                                info_for_logger="_train_step_")
            if self.CONFIG.lr_scheduler == "step":
                self.scheduler.step()

            top1_avg = self._validate(model, val_loader, epoch)
            if best_top1 < top1_avg:
                logging.info("Best top1 acc by now. Save model")
                best_top1 = top1_avg
            save(model, self.optimizer, self.CONFIG.path_to_save_model)

        # Training generator
        best_loss = 10000.0
        best_top1 = 0
        tau = 5
        for epoch in range(self.warmup_epochs, self.search_epochs):
            logging.info("Start to train for search epoch {}".format(epoch))
            logging.info("Tau: {}".format(tau))
            self._generator_training_step(
                generator,
                model,
                val_loader,
                epoch,
                tau,
                info_for_logger="_gen_train_step")

            top1_avg, _ = self.generator_validate(
                generator, model, val_loader, epoch, info_for_logger="_gen_val_step_", target_hardware_constraint=(
                    self.CONFIG.low_flops + self.CONFIG.high_flops) / 2)
            evaluate_metric, total_loss, kendall_tau = evaluate_generator(
                generator, self.prior_pool, self.lookup_table, self.CONFIG, self.device)

            logging.info("Total loss : {}".format(total_loss))
            if best_loss > total_loss:
                logging.info(
                    "Best loss by now: {} Tau : {}.Save model".format(
                        total_loss, kendall_tau))
                best_loss = total_loss
                save_generator_evaluate_metric(
                    evaluate_metric, self.CONFIG.path_to_generator_eval)
                save(
                    generator,
                    self.g_optimizer,
                    self.CONFIG.path_to_save_generator)
            if top1_avg > best_top1 and total_loss < 0.4:
                logging.info(
                    "Best top1-avg by now: {}.Save model".format(top1_avg))
                best_top1 = top1_avg
                save(
                    generator,
                    self.g_optimizer,
                    self.CONFIG.path_to_best_avg_generator)

            tau *= self.CONFIG.tau_decay
        logging.info("Best loss: {}".format(best_loss))
        save(generator, self.g_optimizer, self.CONFIG.path_to_fianl_generator)

    def train_loop(self, train_loader, test_loader, model):
        best_top1 = 0.0
        for epoch in range(self.epochs):
            logging.info(
                "Learning Rate: {:.4f}".format(
                    self.optimizer.param_groups[0]["lr"]))
            self.writer.add_scalar(
                "learning_rate/weights",
                self.optimizer.param_groups[0]["lr"],
                epoch)
            logging.info("Start to train for epoch {}".format(epoch))

            self._training_step(
                model,
                train_loader,
                epoch,
                info_for_logger="_train_step_",
                scratch=True)
            if self.CONFIG.lr_scheduler == "step":
                self.scheduler.step()

            top1_avg = self._validate(model, test_loader, epoch, scratch=True)
            if best_top1 < top1_avg:
                logging.info("Best top1 acc by now. Save model")
                best_top1 = top1_avg
                save(model, self.optimizer, self.CONFIG.path_to_save_scratch)

        logging.info("The Best top1 acc : {}".format(best_top1))
        return best_top1

    def _get_target_hardware_constraint(self, hardware_constraint=None):
        """
        Get target hardware constraint. If the hardware constraint was given, then wrap as the torch tesor.
        """
        if hardware_constraint is None:
            target_hardware_constraint = self.hardware_pool[self.hardware_index] + random.random(
            ) - 0.5
            target_hardware_constraint = torch.tensor(
                target_hardware_constraint, dtype=torch.float32).view(-1, 1)
            self.hardware_index += 1

            if self.hardware_index == len(self.hardware_pool):
                self.hardware_index = 0
                random.shuffle(self.hardware_pool)
        else:
            target_hardware_constraint = torch.tensor(
                hardware_constraint, dtype=torch.float32).view(-1, 1)

        return target_hardware_constraint

    def _get_arch_param(self, generator, target_hardware_constraint=None):
        """
        Given the target hardware constraint as the input of generator. The generator output the architecture parameter.
        """
        hardware_constraint = target_hardware_constraint.to(self.device)
        logging.info("Target flops : {}".format(hardware_constraint.item()))

        prior = self.prior_pool.get_prior(hardware_constraint.item())
        prior = prior.to(self.device)

        normalize_hardware_constraint = min_max_normalize(
            self.CONFIG.high_flops, self.CONFIG.low_flops, hardware_constraint)

        arch_param = generator(prior, normalize_hardware_constraint)

        return arch_param

    def set_arch_param(self, model, arch_param=None, tau=None):
        """
        Set the architecture parameter into the supernet
        """
        if tau is not None:
            arch_param = self.lookup_table.calculate_block_probability(
                arch_param, tau)
        else:
            arch_param = self.lookup_table.get_validation_arch_param(
                arch_param)

        arch_param = arch_param.to(self.device)
        model.module.set_arch_param(arch_param)
        return arch_param

    def _generator_training_step(
            self,
            generator,
            model,
            loader,
            epoch,
            tau,
            info_for_logger=""):
        start_time = time.time()
        generator.train()
        model.eval()

        for step, (X, y) in enumerate(loader):
            self.g_optimizer.zero_grad()
            target_hardware_constraint = self._get_target_hardware_constraint()

            arch_param = self._get_arch_param(
                generator, target_hardware_constraint)
            arch_param = self.set_arch_param(model, arch_param, tau)

            flops = self.lookup_table.get_model_flops(arch_param)
            logging.info("Generate model flops : {}".format(flops))

            hc_loss = cal_hc_loss(
                flops.cuda(),
                target_hardware_constraint.item(),
                self.CONFIG.alpha,
                self.CONFIG.loss_penalty)

            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)
            N = X.shape[0]

            outs = model(X, True)

            ce_loss = self.criterion(outs, y)
            loss = ce_loss + hc_loss
            logging.info("HC loss : {}".format(hc_loss))
            loss.backward()

            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

            self._intermediate_stats_logging(
                outs,
                y,
                loss,
                step,
                epoch,
                N,
                len_loader=len(loader),
                val_or_train="Train",
                hc_losses=hc_loss)
        self._epoch_stats_logging(
            start_time=start_time,
            epoch=epoch,
            val_or_train="train")
        for avg in [self.top1, self.top5, self.losses, self.hc_losses]:
            avg.reset()

    def generator_validate(
            self,
            generator,
            model,
            loader,
            epoch,
            target_hardware_constraint=None,
            arch_param=None,
            info_for_logger=""):
        if generator is not None:
            generator.eval()
        model.eval()
        start_time = time.time()

        if arch_param is None:
            if target_hardware_constraint is None:
                target_hardware_constraint = self._get_target_hardware_constraint()
                arch_param = self._get_arch_param(
                    generator, target_hardware_constraint)
                arch_param = self.set_arch_param(
                    model, arch_param)  # Validate architecture parameter

            else:
                target_hardware_constraint = self._get_target_hardware_constraint(
                    target_hardware_constraint)
                arch_param = self._get_arch_param(
                    generator, target_hardware_constraint)
                arch_param = self.set_arch_param(
                    model, arch_param)  # Validate architecture parameter
        else:
            arch_param = self.set_arch_param(model, arch_param)

        flops = self.lookup_table.get_model_flops(arch_param)
        logging.info("Generate model flops : {}".format(flops))

        hc_loss = cal_hc_loss(
            flops.cuda(),
            target_hardware_constraint.item(),
            self.CONFIG.alpha,
            self.CONFIG.loss_penalty)

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(
                    self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True)
                N = X.shape[0]

                outs = model(X, True)
                loss = self.criterion(outs, y)

                self._intermediate_stats_logging(
                    outs,
                    y,
                    loss,
                    step,
                    epoch,
                    N,
                    len_loader=len(loader),
                    val_or_train="Valid",
                    hc_losses=hc_loss)

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(
            start_time=start_time,
            epoch=epoch,
            val_or_train="val")
        self.writer.add_scalar("train_vs_val/" + "val" + "_hc_", flops, epoch)
        for avg in [self.top1, self.top5, self.losses, self.hc_losses]:
            avg.reset()

        return top1_avg, flops.item()

    def _training_step(
            self,
            model,
            loader,
            epoch,
            info_for_logger="",
            scratch=False):
        model.train()
        start_time = time.time()
        self.optimizer.zero_grad()
        if not scratch:
            model.module.set_training_order(True)

        for step, (X, y) in enumerate(loader):
            X, y = X.to(
                self.device, non_blocking=True), y.to(
                self.device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()
            if not scratch:
                model.module.set_training_order(True)
                for i in range(4):
                    outs = model(X)
                    loss = self.criterion(outs, y)
                    loss.backward()
                    model.module.set_training_order()
            else:
                outs = model(X)
                loss = self.criterion(outs, y)
                loss.backward()

            self.optimizer.step()
            if self.CONFIG.lr_scheduler == "cosine":
                self.scheduler.step()

            self._intermediate_stats_logging(
                outs,
                y,
                loss,
                step,
                epoch,
                N,
                len_loader=len(loader),
                val_or_train="Train")
        self._epoch_stats_logging(
            start_time=start_time,
            epoch=epoch,
            info_for_logger=info_for_logger,
            val_or_train="Train")
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()

    def _validate(self, model, loader, epoch, scratch=False):
        model.eval()
        start_time = time.time()
        if not scratch:
            model.module.set_training_order(True)

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]

                outs = model(X)
                if not scratch:
                    model.module.set_training_order()

                loss = self.criterion(outs, y)
                self._intermediate_stats_logging(
                    outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(
            start_time=start_time,
            epoch=epoch,
            val_or_train="val")
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()

        return top1_avg

    def _epoch_stats_logging(
            self,
            start_time,
            epoch,
            val_or_train,
            info_for_logger=""):
        self.writer.add_scalar(
            "train_vs_val/" +
            val_or_train +
            "_hc_loss" +
            info_for_logger,
            self.hc_losses.get_avg(),
            epoch)
        self.writer.add_scalar(
            "train_vs_val/" +
            val_or_train +
            "_loss" +
            info_for_logger,
            self.losses.get_avg(),
            epoch)
        self.writer.add_scalar(
            "train_vs_val/" +
            val_or_train +
            "_top1" +
            info_for_logger,
            self.top1.get_avg(),
            epoch)
        self.writer.add_scalar(
            "train_vs_val/" +
            val_or_train +
            "_top5" +
            info_for_logger,
            self.top5.get_avg(),
            epoch)

        top1_avg = self.top1.get_avg()
        logging.info(info_for_logger +
                     val_or_train +
                     ":[{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(epoch +
                                                                          1, self.epochs, top1_avg, time.time() -
                                                                          start_time))

    def _intermediate_stats_logging(
            self,
            outs,
            y,
            loss,
            step,
            epoch,
            N,
            len_loader,
            val_or_train,
            hc_losses=None):
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)

        if hc_losses is not None:
            self.hc_losses.update(hc_losses.item(), 1)

        if (step > 1 and step %
                self.CONFIG.print_freq == 0) or step == len_loader - 1:
            logging.info(
                val_or_train +
                ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} HC Loss {:.3f}"
                "Prec@(1, 3) ({:.1%}, {:.1%})".format(
                    epoch +
                    1,
                    self.epochs,
                    step,
                    len_loader -
                    1,
                    self.losses.get_avg(),
                    self.hc_losses.get_avg(),
                    self.top1.get_avg(),
                    self.top5.get_avg()))
