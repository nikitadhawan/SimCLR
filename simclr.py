import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from loss import soft_nn_loss, pairwise_euclid_distance
from pytorch_stats_loss import WassersteinLoss

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, stealing=False, model_to_steal=None, victim_mlp=None, logdir='simclr_cifar10/', *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        if not stealing:
            self.mlp = kwargs['mlp'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir='runs/'+logdir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.stealing = stealing
        if self.stealing:
            self.soft_nn_loss = soft_nn_loss
            self.mse = torch.nn.MSELoss().to(self.args.device)
            self.wasserstein = WassersteinLoss().to(self.args.device)
            self.model_to_steal = model_to_steal.to(self.args.device)
            self.victim_mlp = victim_mlp.to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, watermark_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            watermark_accuracy = 0
            for images, _ in tqdm(watermark_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits = self.mlp(features)
                    # labels = torch.cat([torch.tensor([0, 1]) for _ in range(args.batch_size)], dim=0).to(self.args.device)
                    labels = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0).long().to(self.args.device)
                    loss = self.criterion(logits, labels)
                    w_top1 = accuracy(logits, labels, topk=(1,))
                    watermark_accuracy += w_top1[0]

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tWatermark accuracy: {w_top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'mlp_state_dict': self.mlp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        watermark_accuracy = 0
        for counter, (x_batch, _) in enumerate(watermark_loader):
            x_batch = torch.cat(x_batch, dim=0)
            x_batch = x_batch.to(device)
            logits = mlp(model(x_batch))
            y_batch = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0).long().to(self.args.device)
            # y_batch = torch.cat([torch.tensor([0, 1]) for _ in range(self.args.batch_size)], dim=0).to(device)
            top1 = accuracy(logits, y_batch, topk=(1,))
            watermark_accuracy += top1[0]
        watermark_accuracy /= (counter+1)
        logging.info(f"Watermark accuracy is {watermark_accuracy}.")
        
        
    def steal(self, train_loader, watermark_loader, num_queries):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR stealing for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        total_queries = 0

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    query_features = self.model_to_steal(images)
                    features = self.model(images)
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = self.info_nce_loss(all_features)
                    # loss = soft_nn_loss(self.args, all_features, pairwise_euclid_distance, 100)
                    loss = self.mse(features, query_features)
                    # loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
                total_queries += len(images)
                if total_queries >= num_queries:
                    break;

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Stealing has finished.")
        # save model checkpoints
        checkpoint_name = 'stolen_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Stolen model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        watermark_accuracy = 0
        for counter, (x_batch, _) in enumerate(watermark_loader):
            x_batch = torch.cat(x_batch, dim=0)
            x_batch = x_batch.to(self.args.device)
            logits = self.victim_mlp(self.model(x_batch))
            y_batch = torch.cat([torch.zeros(self.args.batch_size), torch.ones(self.args.batch_size)], dim=0).long().to(self.args.device)
            top1 = accuracy(logits, y_batch, topk=(1,))
            watermark_accuracy += top1[0]
        watermark_accuracy /= (counter+1)
        logging.info(f"Watermark accuracy is {watermark_accuracy}.")


