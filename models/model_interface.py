import os
import math
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import pytorch_lightning as pl

from utils import instantiate_from_config
from data.data_utils import find_class_imbalance
from utils.analysis import accuracy, binary_accuracy, js_div

class DCBMInterface(pl.LightningModule):
    def __init__(self, 
                 model_config, 
                 use_attr, 
                 no_img, 
                 data_dir, 
                 n_attributes, 
                 batch_size, 
                 max_epochs, 
                 dataset):
        super().__init__()
        self.save_hyperparameters()
        self.values = dict() # log_dict

        self.dcbm = instantiate_from_config(model_config.dcbm_config)

        self.use_attr = use_attr
        self.no_img = no_img
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_attributes = n_attributes
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.save_dir = model_config.save_dir
        self.optimizer = model_config.optimizer
        self.concept_percent = model_config.concept_percent
        self.lr = model_config.lr
        self.min_lr = model_config.min_lr
        self.lr_decay_size = model_config.lr_decay_size
        self.weight_decay = model_config.weight_decay
        self.weighted_loss = model_config.weighted_loss
        self.scheduler_step = model_config.scheduler_step
        self.attr_loss_weight = model_config.attr_loss_weight
        self.embs_weight = model_config.embs_weight
        self.use_embs = model_config.use_embs
        self.use_aux = model_config.dcbm_config.params.use_aux

        self.stop_epoch = int(math.log(self.min_lr / self.lr) / math.log(self.lr_decay_size)) * self.scheduler_step
        print(f"Schedule stop epochs: {self.stop_epoch}")

    def preprocessing(self):
        # compute attribute imbalance
        imbalance = None
        if self.dataset in ["CUB", "Derm7pt"]:
            # CUB, Derm7pt
            if self.use_attr and not self.no_img and self.weighted_loss:
                train_data_path = os.path.join(self.data_dir, 'train.pkl')
                if self.weighted_loss == 'multiple':
                    imbalance = find_class_imbalance(train_data_path, True, concept_percent=self.concept_percent)
                else:
                    imbalance = find_class_imbalance(train_data_path, False, concept_percent=self.concept_percent)
        elif self.dataset in ['celeba']:
            # celeba
            with open(os.path.join(self.save_dir, 'celeba_imbalance.pth'), "rb") as f:
                imbalance = pkl.load(f)

        # define loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.attr_criterion = [] #separate criterion (loss function) for each attribute
        if self.weighted_loss:
            assert(imbalance is not None)
            print("Use weighted loss.")
            for ratio in imbalance:
                self.attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio])).to(self.device))
        else:
            for i in range(n_attributes):
                self.attr_criterion.append(torch.nn.CrossEntropyLoss())

    def on_fit_start(self):
        self.preprocessing()

    def on_test_start(self):
        self.preprocessing()

    def forward(self, batch, trans=False):
        if self.dataset in ["CUB", "Derm7pt"]:
            inputs, labels, attr_labels = batch
        elif self.dataset in ['celeba']:
            inputs, (labels, attr_labels) = batch
        if self.n_attributes > 1:
            attr_labels = [i.long() for i in attr_labels]
            attr_labels = torch.stack(attr_labels).t()  # .float() #N x 312
            if self.dataset in ['celeba']:
                attr_labels = attr_labels.T
        else:
            if isinstance(attr_labels, list):
                attr_labels = attr_labels[0]
            attr_labels = attr_labels.unsqueeze(1)
        attr_labels_var = attr_labels.float().to(self.device)
        inputs_var = inputs.to(self.device)
        labels_var = labels.to(self.device)
        return inputs_var, attr_labels_var, labels_var

    def training_step(self, batch, batch_idx):
        self.dcbm.train()

        inputs_var, attr_labels_var, labels_var = self(batch)

        if self.use_aux:
            outputs, aux_outputs = self.dcbm(inputs_var)
            mixed_outputs = outputs[1] + outputs[3]
            losses = []
            # cross_entropy
            loss_main = 1.0 * self.criterion(mixed_outputs, labels_var) + 0.4 * self.criterion(aux_outputs[1] + aux_outputs[3], labels_var)
            losses.append(loss_main)
            # explicit mapping

            loss_concept = [
                self.attr_loss_weight * (
                    1.0 * self.attr_criterion[i](outputs[2][:, i].float(), attr_labels_var[:, i])
                    + 0.4 * self.attr_criterion[i](aux_outputs[2][:, i], attr_labels_var[:, i])
                )
                for i in range(len(self.attr_criterion))
            ]
        else:
            outputs = self.dcbm(inputs_var)
            mixed_outputs = outputs[1] + outputs[3]
            losses = []
            loss_main = 1.0 * self.criterion(mixed_outputs, labels_var)
            losses.append(loss_main)
            # explicit mapping
            loss_concept = [self.attr_loss_weight * (
                        1.0 * self.attr_criterion[i](outputs[2][:, i].float().to(self.device),
                                                attr_labels_var[:, i])) for i in range(len(self.attr_criterion))]
        
        loss_concept = sum(loss_concept)
        losses.append(loss_concept)
        self.log_util(loss_main, 'loss_main')
        self.log_util(loss_concept, 'loss_concept')
        
        # implicit embedding
        if self.use_embs:
            loss_implicit = js_div(outputs[1], mixed_outputs)
            loss_penalty = self.embs_weight * loss_implicit
            loss_explicit = js_div(outputs[3], mixed_outputs)

            self.log_util(loss_penalty, 'loss_penalty')
            losses.append(loss_penalty)
            self.log_util(loss_implicit, 'loss_implicit')
            self.log_util(loss_explicit, 'loss_explicit')

        sigmoid_outputs = torch.nn.Sigmoid()(outputs[2])
        acc_attr = binary_accuracy(sigmoid_outputs, attr_labels_var)
        self.log_util(acc_attr, 'train_acc_attr')
        acc = accuracy(mixed_outputs, labels_var, topk=(1,))  # only care about class prediction accuracy
        self.log_util(acc[0], 'train_acc_label')
        class_implicit_acc = accuracy(outputs[3], labels_var, topk=(1,))
        class_explicit_acc = accuracy(outputs[1], labels_var, topk=(1,))
        self.log_util(class_implicit_acc[0], 'train_implicit_acc_label')
        self.log_util(class_explicit_acc[0], 'train_explicit_acc_label')

        if self.use_embs:
            total_loss = sum(losses[:-1]) / (1 + self.attr_loss_weight * self.n_attributes) + losses[-1]
        else:
            total_loss = sum(losses) / (1 + self.attr_loss_weight * self.n_attributes)
        self.log_util(total_loss, 'train_loss')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.dcbm.eval()

        inputs_var, attr_labels_var, labels_var = self(batch)

        outputs = self.dcbm(inputs_var)
        mixed_outputs = outputs[1] + outputs[3]
        losses = []
        loss_main = 1.0 * self.criterion(mixed_outputs, labels_var)
        losses.append(loss_main)
        # explicit mapping
        loss_concept = [
            self.attr_loss_weight * (
                1.0 * self.attr_criterion[i](outputs[2][:, i].float(), attr_labels_var[:, i])
            )
            for i in range(len(self.attr_criterion))
        ]
        loss_concept = sum(loss_concept)
        losses.append(loss_concept)

        sigmoid_outputs = torch.nn.Sigmoid()(outputs[2])
        acc_attr = binary_accuracy(sigmoid_outputs, attr_labels_var)
        self.log_util(acc_attr, 'val_acc_attr')
        acc = accuracy(mixed_outputs, labels_var, topk=(1,))  # only care about class prediction accuracy
        self.log_util(acc[0], 'val_acc_label')
        class_implicit_acc = accuracy(outputs[3], labels_var, topk=(1,))
        class_explicit_acc = accuracy(outputs[1], labels_var, topk=(1,))
        self.log_util(class_implicit_acc[0], 'val_implicit_acc_label')
        self.log_util(class_explicit_acc[0], 'val_explicit_acc_label')

        total_loss = sum(losses) / (1 + self.attr_loss_weight * self.n_attributes)
        self.log_util(total_loss, 'val_loss')
        return total_loss

    def test_step(self, batch, batch_idx):
        self.dcbm.eval()

        inputs_var, attr_labels_var, labels_var = self(batch)

        outputs = self.dcbm(inputs_var)
        mixed_outputs = outputs[1] + outputs[3]
        losses = []
        loss_main = 1.0 * self.criterion(mixed_outputs, labels_var)
        losses.append(loss_main)
        # explicit mapping
        loss_concept = [
            self.attr_loss_weight * (
                1.0 * self.attr_criterion[i](outputs[2][:, i].float(), attr_labels_var[:, i])
            )
            for i in range(len(self.attr_criterion))
        ]
        loss_concept = sum(loss_concept)
        losses.append(loss_concept)

        sigmoid_outputs = torch.nn.Sigmoid()(outputs[2])
        acc_attr = binary_accuracy(sigmoid_outputs, attr_labels_var)
        self.log_util(acc_attr, 'test_acc_attr')
        acc = accuracy(mixed_outputs, labels_var, topk=(1,))  # only care about class prediction accuracy
        self.log_util(acc[0], 'test_acc_label')
        class_implicit_acc = accuracy(outputs[3], labels_var, topk=(1,))
        class_explicit_acc = accuracy(outputs[1], labels_var, topk=(1,))
        self.log_util(class_implicit_acc[0], 'test_implicit_acc_label')
        self.log_util(class_explicit_acc[0], 'test_explicit_acc_label')

        total_loss = sum(losses) / (1 + self.attr_loss_weight * self.n_attributes)
        self.log_util(total_loss, 'test_loss')
        return total_loss

    def log_util(self, loss, name='loss'):
        self.values[name] = loss
        self.log_dict(self.values, logger=True, prog_bar=True, on_step=False, on_epoch=True, 
                      batch_size=self.batch_size)
        
    def on_train_epoch_end(self):
        if self.current_epoch <= self.stop_epoch:
            self.scheduler.step(self.current_epoch)
        
    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dcbm.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.dcbm.parameters()), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.dcbm.parameters()), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamp':
            from adamp import AdamP
            optimizer = AdamP(filter(lambda p: p.requires_grad, self.dcbm.parameters()), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        
        if self.optimizer == 'adamp':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.1)
        
        return [optimizer], []
