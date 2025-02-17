import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
            for i in range(self.n_attributes):
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
            attr_labels = torch.stack(attr_labels).t()
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

class MINEInterface(pl.LightningModule):
    def __init__(self, 
                 model_config, 
                 mine_steps, 
                 dataset, 
                 n_attributes, 
                 eta):
        super().__init__()
        self.save_hyperparameters()
        self.values = dict() # log_dict

        self.d_f = instantiate_from_config(model_config.decouple_config)
        self.mine = instantiate_from_config(model_config.mine_config)
        self.mine_steps = mine_steps
        self.dataset = dataset
        self.n_attributes = n_attributes
        self.eta = eta

        self.decouple_lr = model_config.decouple_lr
        self.mine_lr = model_config.mine_lr
        self.decouple_optimizer = model_config.decouple_optimizer
        self.mine_optimizer = model_config.mine_optimizer

        self.criterion = nn.CrossEntropyLoss()

        self.automatic_optimization = False # switch off the automatic optimization

    def training_step(self, batch, batch_idx):
        decouple_opt, mine_opt = self.optimizers()
        # Extract batch data
        c_explicit = batch['c_explicit']
        c_implicit = batch['c_implicit']
        c_truth = batch['c_truth']
        y_truth = batch['y_truth']
        W_explicit = batch['W_explicit'][0]  # Take first since all same
        b_explicit = batch['b_explicit'][0]
        W_implicit = batch['W_implicit'][0]
        b_implicit = batch['b_implicit'][0]

        with torch.no_grad():
            mapping_explicit = self.d_f(c_explicit)
            residual_implicit = c_implicit - mapping_explicit

        for _ in range(self.mine_steps):
            mine_opt.zero_grad()
            loss = torch.zeros(1, device=self.device)
            for _ in range(5):
                loss = loss - 1/5 * self.mine(c_explicit, residual_implicit)
            self.manual_backward(loss)
            mine_opt.step()

        # Decoupling optimization
        mi = self.mine.eval()(c_explicit, c_implicit - self.d_f(c_explicit))
        regularization = torch.mean(self.d_f(c_explicit)**2)
        loss = mi + self.eta * regularization

        # Calculate new loss with intervention
        new_loss = self.calculate_intervention_loss(c_explicit, c_implicit, y_truth, c_truth,
                                                  W_explicit, b_explicit, W_implicit, b_implicit)
        
        total_loss = loss + new_loss
        
        decouple_opt.zero_grad()
        self.manual_backward(total_loss)
        decouple_opt.step()

        # Log metrics
        self.log('train_mi', mi.item(), prog_bar=True)
        self.log('train_reg', regularization.item(), prog_bar=True)
        self.log('train_new_loss', new_loss.item(), prog_bar=True)
        self.log('train_total_loss', total_loss.item(), prog_bar=True)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        c_explicit = batch['c_explicit']
        c_implicit = batch['c_implicit']
        c_truth = batch['c_truth']
        y_explicit = batch['y_explicit']
        y_implicit = batch['y_implicit']
        y_truth = batch['y_truth']
        W_explicit = batch['W_explicit'][0]
        b_explicit = batch['b_explicit'][0]
        W_implicit = batch['W_implicit'][0]
        b_implicit = batch['b_implicit'][0]

        acc_origin, acc_int, acc_mine = self.calculate_accuracies(
            c_explicit, c_implicit, y_explicit, y_implicit, y_truth, c_truth,
            W_explicit, b_explicit, W_implicit, b_implicit
        )

        self.log('test_acc_origin', acc_origin, on_epoch=True)
        self.log('test_acc_int', acc_int, on_epoch=True)
        self.log('test_acc_mine', acc_mine, on_epoch=True)

        return {'test_acc_origin': acc_origin, 
                'test_acc_int': acc_int, 
                'test_acc_mine': acc_mine}

    def calculate_intervention_loss(self, c_explicit, c_implicit, y_truth, c_truth,
                                  W_explicit, b_explicit, W_implicit, b_implicit):
        mapping_explicit = self.d_f(c_explicit)
        residual_implicit = c_implicit - mapping_explicit
        
        # Create intervention mask based on dataset
        if self.dataset == 'CUB':
            cut = torch.ones((c_explicit.shape[0], self.n_attributes), device=self.device)
            cut[:, 11:] = 0
        elif self.dataset == 'Derm7pt':
            cut = torch.zeros((c_explicit.shape[0], 8), device=self.device)
            cut[:, 1:4] = 1
        else:
            cut = torch.ones_like(c_explicit)
            
        # Calculate interventions
        quantile95 = torch.quantile(c_explicit, 0.95, dim=0)
        quantile05 = torch.quantile(c_explicit, 0.05, dim=0)
        
        c_explicit_new = (torch.sign(c_explicit * cut) == torch.sign(c_truth * cut)) * c_explicit + \
                        (torch.sign(c_explicit * cut) > torch.sign(c_truth * cut)) * quantile05 + \
                        (torch.sign(c_explicit * cut) < torch.sign(c_truth * cut)) * quantile95

        c_implicit_new = self.d_f(c_explicit_new) + residual_implicit
        mine_y_new = torch.mm(c_explicit_new, W_explicit.t()) + b_explicit + \
                     b_implicit + torch.mm(c_implicit_new, W_implicit.t())

        return self.criterion(mine_y_new, y_truth)
    
    def calculate_accuracies(self, c_explicit, c_implicit, y_explicit, y_implicit, 
                           y_truth, c_truth, W_explicit, b_explicit, W_implicit, b_implicit):
        mapping_test_explicit = self.d_f(c_explicit)
        residual_test_implicit = c_implicit - mapping_test_explicit
        
        # Create intervention mask based on dataset
        if self.dataset == 'CUB':
            cut = torch.ones((c_explicit.shape[0], self.n_attributes), device=self.device)
            cut[:, 11:] = 0
        elif self.dataset == 'Derm7pt':
            cut = torch.zeros((c_explicit.shape[0], 8), device=self.device)
            cut[:, 1:4] = 1
        else:
            cut = torch.ones_like(c_explicit)

        # Calculate interventions
        quantile95 = torch.quantile(c_explicit, 0.95, dim=0)
        quantile05 = torch.quantile(c_explicit, 0.05, dim=0)
        
        c_explicit_new = (torch.sign(c_explicit * cut) == torch.sign(c_truth * cut)) * c_explicit + \
                        (torch.sign(c_explicit * cut) > torch.sign(c_truth * cut)) * quantile05 + \
                        (torch.sign(c_explicit * cut) < torch.sign(c_truth * cut)) * quantile95

        test_y_new = torch.mm(c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + \
                     torch.mm(c_implicit, W_implicit.t())
                     
        acc_origin = ((y_implicit + y_explicit).max(axis=1)[1] == y_truth).sum() / y_truth.shape[0] * 100
        acc_int = (test_y_new.max(axis=1)[1] == y_truth).sum() / y_truth.shape[0] * 100
        
        c_implicit_new = self.d_f(c_explicit_new).detach() + residual_test_implicit
        mine_y_new = torch.mm(c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + \
                     torch.mm(c_implicit_new, W_implicit.t())
        acc_mine = (mine_y_new.max(axis=1)[1] == y_truth).sum() / y_truth.shape[0] * 100
        
        return acc_origin, acc_int, acc_mine
    
    def configure_optimizers(self):
        if self.decouple_optimizer == 'Adam':
            decouple_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.d_f.parameters()), lr=self.decouple_lr)
        elif self.decouple_optimizer == 'RMSprop':
            decouple_opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.d_f.parameters()), lr=self.decouple_lr)
        elif self.decouple_optimizer == 'SGD':
            decouple_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.d_f.parameters()), lr=self.decouple_lr)
        
        if self.mine_optimizer == 'Adam':
            mine_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mine.parameters()), lr=self.mine_lr)
        elif self.mine_optimizer == 'RMSprop':
            mine_opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.mine.parameters()), lr=self.mine_lr)
        elif self.mine_optimizer == 'SGD':
            mine_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.mine.parameters()), lr=self.mine_lr)

        return [decouple_opt, mine_opt], []

class RECInterface(pl.LightningModule):
    def __init__(self, d_f, alpha, rec_lr, rec_steps=2000):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_f = d_f
        self.rec_steps = rec_steps
        self.alpha = alpha
        self.rec_lr = rec_lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, c_explicit):
        return self.d_f(c_explicit)

    def training_step(self, batch, batch_idx):
        self.c_explicit = batch['c_explicit']
        self.c_implicit = batch['c_implicit']
        self.c_truth = batch['c_truth']
        self.y_explicit = batch['y_explicit']
        self.y_implicit = batch['y_implicit']
        self.y_truth = batch['y_truth']
        self.W_explicit = batch['W_explicit'][0]
        self.b_explicit = batch['b_explicit'][0]
        self.W_implicit = batch['W_implicit'][0]
        self.b_implicit = batch['b_implicit'][0]

        # Initialize variables for rectification
        c0 = self.c_explicit.clone().detach()
        c_explicit_var = c0.clone().detach().requires_grad_(True)
        c_implicit_var = self.c_implicit.clone().detach()
        
        # Optimize explicit concepts
        optimizer = torch.optim.Adam([c_explicit_var], lr=self.rec_lr)

        for step in tqdm(range(self.rec_steps)):
            mapping_explicit = self.d_f(c_explicit_var)
            residual_implicit = (c_implicit_var - mapping_explicit).detach()
            
            # Combined prediction
            pred = (c_explicit_var.matmul(self.W_explicit.T) + self.b_explicit + 
                   self.b_implicit + (mapping_explicit + residual_implicit).matmul(self.W_implicit.T))
            l1 = F.cross_entropy(pred, self.y_truth, reduction='none')
            l2 = self.alpha * (c_explicit_var - c0) ** 2 * c_explicit_var.shape[0]
            loss = l1 + l2.sum(1)
            if step > 10 and hasattr(self, 'prev_loss'):
                mask = torch.abs(self.prev_loss - loss.detach()) > 1e-4
                self.prev_loss = loss.detach()
                loss = loss * mask
            else:
                self.prev_loss = loss.detach()
            if loss.max() == 0:
                break

            # Calculate losses
            loss = loss.mean()
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.c_explicit_var = c_explicit_var.detach()
        return torch.tensor(0.0, device=self.device).requires_grad_(True) # fake loss (metrics['loss'] must be required in pytorch_lightning)

    def test_step(self, batch, batch_idx):
        metrics = {}

        acc_list = []
        for i in range(len(self.y_explicit)):
            acc = accuracy(self.y_explicit[i].unsqueeze(0) + self.y_implicit[i].unsqueeze(0), 
                           self.y_truth[i].unsqueeze(0), topk=[1])[0][0].cpu().numpy()
            acc_list.append(acc)
        # acc_list = torch.tensor(np.array(acc_list), device=self.device).unsqueeze(1)
        wrong_list = (1 - np.array(acc_list) / 100).astype(bool)

        original_concepts = torch.sigmoid(self.c_explicit[wrong_list])
        rectified_concepts = torch.sigmoid(self.c_explicit_var[wrong_list])

        self.c_truth = self.c_truth/2 + 0.5
        self.c_truth = self.c_truth[wrong_list]

        # Calculate metrics
        # metrics['auc'] = roc_auc_score(
        #     self.c_truth.cpu().numpy(), 
        #     rectified_concepts.cpu().numpy()
        # )
        
        # Convert to binary predictions
        orig_binary = (original_concepts > 0.5).float()
        # pred_binary = (rectified_concepts * (1-acc_list) + original_concepts * acc_list > 0.5).float()
        pred_binary = (rectified_concepts > 0.5).float()
       
        metrics['original_accuracy'] = binary_accuracy(
            orig_binary, 
            self.c_truth
        ).mean()

        metrics['rectification_accuracy'] = binary_accuracy(
            pred_binary, 
            self.c_truth
        ).mean()
            
        self.log_dict(metrics)
        self.metrics = metrics
        return metrics

    def configure_optimizers(self):
        return None

    # inference each sample

    # def rectify_sample(self, anchor_idx):
    #     # Skip if prediction is already rectified
    #     if accuracy(self.y_explicit[anchor_idx].unsqueeze(0) + self.y_implicit[anchor_idx].unsqueeze(0), self.y_truth[anchor_idx].unsqueeze(0), topk=[1])[0][0].cpu().numpy() != 0:
    #         return None
            
    #     # Initialize variables for rectification
    #     c0 = self.c_explicit.clone().detach()
    #     c_explicit_var = c0[anchor_idx].clone().detach().requires_grad_(True)
    #     # c_explicit_var = torch.autograd.Variable(c0[anchor_idx].clone().detach())
    #     c_implicit_var = self.c_implicit[anchor_idx].clone().detach()
        
    #     # Optimize explicit concepts
    #     optimizer = torch.optim.Adam([c_explicit_var], lr=self.rec_lr)
        
    #     for step in range(self.rec_steps):
    #         mapping_explicit = self.d_f(c_explicit_var.unsqueeze(0))
    #         residual_implicit = (c_implicit_var - mapping_explicit[0]).detach()
            
    #         # Combined prediction
    #         pred = (c_explicit_var.matmul(self.W_explicit.T) + self.b_explicit + 
    #                self.b_implicit + (mapping_explicit[0] + residual_implicit).matmul(self.W_implicit.T))
            
    #         # Calculate losses
    #         l1 = self.criterion(pred.unsqueeze(0), self.y_truth[anchor_idx].unsqueeze(0))
    #         l2 = self.alpha * torch.norm(c_explicit_var - c0[anchor_idx]) ** 2
    #         loss = l1 + l2
    #         # Optimization step
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         if step > 10 and hasattr(self, 'prev_loss') and abs(self.prev_loss - loss.item()) <= 1e-5:
    #             break
    #         self.prev_loss = loss.item()
        
    #     self.loss = loss
    #     return c_explicit_var.detach()

    # def training_step(self, batch, batch_idx):
    #     data = batch
    #     self.c_explicit = data['c_explicit']
    #     self.c_implicit = data['c_implicit']
    #     self.c_truth = data['c_truth']
    #     self.y_explicit = data['y_explicit']
    #     self.y_implicit = data['y_implicit']
    #     self.y_truth = data['y_truth']
    #     self.W_explicit = data['W_explicit'][0]
    #     self.b_explicit = data['b_explicit'][0]
    #     self.W_implicit = data['W_implicit'][0]
    #     self.b_implicit = data['b_implicit'][0]
    #     metrics = {}
    #     rectified_concepts = []
    #     original_concepts = []
    #     ground_truth = []
        
    #     for idx in tqdm(range(len(self.c_explicit))):
    #         result = self.rectify_sample(idx, data)
    #         if result is not None:
    #             original_concepts.append(torch.sigmoid(self.c_explicit[idx]))
    #             rectified_concepts.append(torch.sigmoid(result))
    #             ground_truth.append(self.c_truth[idx])
        
    #     with torch.no_grad():
    #         if rectified_concepts:
    #             original_concepts = torch.stack(original_concepts)
    #             rectified_concepts = torch.stack(rectified_concepts)
    #             ground_truth = torch.stack(ground_truth)
    #             ground_truth = ground_truth/2 + 0.5

    #             metrics['loss'] = torch.tensor(0.0, device=self.device).requires_grad_(True) # fake loss (metrics['loss'] must be required in pytorch_lightning)

    #             # Calculate metrics
    #             metrics['auc'] = roc_auc_score(
    #                 ground_truth.cpu().numpy(), 
    #                 rectified_concepts.cpu().numpy()
    #             )
                
    #             # Convert to binary predictions
    #             pred_binary = (rectified_concepts > 0.5).float()
    #             orig_binary = (original_concepts > 0.5).float()
                
    #             metrics['original_accuracy'] = binary_accuracy(
    #                 orig_binary, 
    #                 ground_truth
    #             ).mean()

    #             metrics['rectification_accuracy'] = binary_accuracy(
    #                 pred_binary, 
    #                 ground_truth
    #             ).mean()
                
    #         self.log_dict(metrics)
    #         self.metrics = metrics
    #     return metrics

    # def on_train_epoch_end(self):
    #     # Print the final metrics
    #     print("\nResults:")
    #     for key, value in self.metrics.items():
    #         print(f"{key}: {value.item():.4f}")

    # def configure_optimizers(self):
    #     # Not needed for testing/inference
    #     return None