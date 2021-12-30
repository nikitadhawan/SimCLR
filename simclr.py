import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(
            torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights,
                      1))
    else:
        return torch.mean(
            torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


class SimCLR(object):

    def __init__(self, stealing=False, victim_model=None, logdir='', loss=None, *args,
                 **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.log_dir = 'runs/' + logdir
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.stealing = stealing
        self.loss = loss
        if os.path.exists(os.path.join(self.log_dir, 'training.log')):
            os.remove(os.path.join(self.log_dir, 'training.log'))
        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'training.log'),
            level=logging.DEBUG)
        if self.stealing:
            self.victim_model = victim_model.to(self.args.device)
        if self.loss == "softce":
            self.criterion = soft_cross_entropy

    def info_nce_loss(self, features):
        n = int(features.size()[0] / self.args.batch_size)
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        if not self.stealing:
            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(
                similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
            self.args.device)
        logits = logits / self.args.temperature  # Do we need temperature scaling for soft CE loss?
        # print("labels", torch.sum(labels))
        # print("logits",logits)
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")

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

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = f'{self.args.dataset}_checkpoint_{self.args.epochs}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(self.log_dir, checkpoint_name))
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.log_dir}.")

    def steal(self, train_loader, num_queries):
        # Note: We use the test set to attack the model.

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR stealing for {self.args.epochs} epochs.")
        logging.info(f"Using loss type: {self.loss}")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")


        for epoch_counter in range(self.args.epochs):
            total_queries = 0
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                # Add augmentations / different querying strategies.
                images = images.to(self.args.device)

                query_features = self.victim_model(images) # victim model representations
                features = self.model(images) # current stolen model representation: 512x128 (512 images, 128 dimensional representation)
                if self.loss == "softce":
                    loss = self.criterion(features, F.softmax(query_features/self.args.temperature, dim=1)) # F.softmax(features, dim=1)
                else:
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = self.info_nce_loss(all_features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1
                total_queries += len(images)
                if total_queries >= num_queries:
                    break

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        logging.info("Stealing has finished.")
        # save model checkpoints
        if self.loss == "softce":
            checkpoint_name = f'stolen_checkpoint_{self.args.epochs}.pth.tar'
        else:
            checkpoint_name = f'stolen_checkpoint_{self.args.epochs}_infonce.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(self.log_dir, checkpoint_name))
        logging.info(
            f"Stolen model checkpoint and metadata has been saved at {self.log_dir}.")
