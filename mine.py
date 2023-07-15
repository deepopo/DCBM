import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils.dataset import load_data
from utils.config import BASE_DIR, N_ATTRIBUTES
from utils.analysis import Logger, AverageMeter, multiclass_metric, accuracy, binary_accuracy, js_div
from sklearn.metrics import roc_auc_score 
import torch.nn as nn
import math
from time import time
import pickle as pkl

import argparse

class MINE(nn.Module):
    def __init__(self, dim_explicit, dim_implicit) :
        super(MINE, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear((dim_explicit + dim_implicit), 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1))

    def forward(self, e_explicit, e_implicit):
        batch_size = e_explicit.size(0)
        tiled_x = torch.cat([e_explicit, e_explicit, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = e_implicit[idx]
        concat_y = torch.cat([e_implicit, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        mi = math.log2(math.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        return mi

    def optimize(self, e_explicit, e_implicit, iters, opt=None):
        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            opt.zero_grad()
            loss = torch.Tensor([0.0]).to(e_explicit.device)
            for _ in range(5):
                loss = loss - 0.2 * self.forward(e_explicit, e_implicit)
            loss.backward()
            opt.step()
        final_mi = self.forward(e_explicit, e_implicit)
        #print(f"Final MI: {final_mi}")


class Decoupling(nn.Module):
    def __init__(self, dim_explicit, dim_implicit):
        super(Decoupling, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_explicit, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, dim_implicit))

    def forward(self, x):
        return self.layers(x)
        
def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-log_dir2', default=None, nargs='+', help='inference log directory')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-seed', type=int, help='seed')

    parser.add_argument('-lr1', type=float, help="learning rate")
    parser.add_argument('-lr2', type=float, help="learning rate")
    parser.add_argument('-cuda_device', default=0, type=int, help='which cuda device to use')
    parser.add_argument('-use_cuda', default=True, type=bool, help='where to use cuda')
    parser.add_argument('-n_attributes', type=int, default=112,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-model_dir', default='', help='choose which model to intervene/rectify')
    parser.add_argument('-model_dir2', default=None, nargs='+', help='choose which model to intervene/rectify')

    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
    parser.add_argument('-inference', action='store_true',
                        help='Whether to infer')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-concept_percent', type=int,
                        choices=[100, 50, 40, 30, 20, 10],
                        default=100,
                        help='Use how many concepts.')
    args = parser.parse_args()
    return args

def run_mine(args, train_c_explicit, train_c_implicit, train_y_explicit, train_y_implicit, train_y_truth, train_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, quantile95, quantile05 
# , test_c_explicit, test_c_implicit, test_y_truth, test_c_truth
):
    d_f = Decoupling(train_c_explicit.shape[1], train_c_implicit.shape[1]).to(args.device)
    opt = torch.optim.Adam(d_f.parameters(), lr=args.lr1)

    iters = 20

    mi_nn = MINE(train_c_explicit.shape[1], train_c_implicit.shape[1]).to(args.device)
    opt_mine = torch.optim.SGD(mi_nn.parameters(), lr=args.lr2)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1
    k = 0
    best_loss3 = 1000
    for epoch in range(args.epochs):
        with torch.no_grad():
            mapping_explicit = d_f(train_c_explicit)
            residual_implicit = train_c_implicit - mapping_explicit
        mi_nn.optimize(train_c_explicit, residual_implicit, iters, opt_mine)
        mi = mi_nn.eval()(train_c_explicit, train_c_implicit - d_f(train_c_explicit))
        
        loss3 = new_loss(train_c_explicit, train_c_implicit, train_y_truth, train_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, criterion, args)
        # loss3 = new_loss(test_c_explicit, test_c_implicit, test_y_truth, test_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, criterion, args)
        
        loss = mi + 1.0 * torch.mean(d_f(train_c_explicit)**2)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        acc_origin, acc_int, acc_mine = acc(train_c_explicit, train_c_implicit, train_y_explicit, train_y_implicit, train_y_truth, train_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, args)

        # acc_origin2, acc_int2, acc_mine2 = acc(test_c_explicit, test_c_implicit, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, args)
       

        if best_loss3 > loss3:
            best_loss3 = loss3
            torch.save(d_f, os.path.join(args.log_dir, 'mine.pth'))
            k = 0
        else:
            k += 1
            
        if k > 1000:
            break
        if epoch % 100 == 0:
            logger.write('iteration: %d, loss: %.4f, mi: %.4f, loss3: %.4f\n' % (epoch, loss.item(), mi.item(), loss3))
            logger.write('TRAIN acc_origin: %.4f, acc_int: %.4f, acc_mine: %.4f\n' % (acc_origin, acc_int, acc_mine))
            # logger.write('TEST acc_origin: %.4f, acc_int: %.4f, acc_mine: %.4f, best_loss: %.4f\n' % (acc_origin2, acc_int2, acc_mine2, best_loss3)) ### TEST
            logger.flush()
    return d_f

def data_gradients(loader, model):
    c_explicit = []
    y = []
    c_truth = []
    y_explicit = []
    y_implicit = []
    y_truth = []
    c_implicit = []
    for data_idx, data in tqdm(enumerate(loader)):
        inputs, labels, attr_labels = data
        inputs_var = torch.autograd.Variable(inputs).to(args.device)
        attr_labels = torch.stack(attr_labels).t()  # N x 312
        outputs = model(inputs_var)
        attr_outputs = outputs[2]
        c_explicit.extend([attr_outputs.detach()])
        y.extend([outputs[1].detach() + outputs[3].detach()])
        y_explicit.extend([outputs[1].detach()])
        y_implicit.extend([outputs[3].detach()])
        c_implicit.extend([outputs[4].detach()])
        y_truth.extend([labels])
        c_truth.extend([attr_labels])
    c_explicit = torch.cat(c_explicit, dim=0)
    c_implicit = torch.cat(c_implicit, dim=0)
    y = torch.cat(y, dim=0)
    y_explicit = torch.cat(y_explicit, dim=0)
    y_implicit = torch.cat(y_implicit, dim=0)
    y_truth = torch.cat(y_truth, dim=0)
    c_truth = torch.cat(c_truth, dim=0)
    c_truth = c_truth * 2 - 1.0
    return c_explicit, c_implicit, y, y_explicit, y_implicit, y_truth, c_truth

def acc(test_c_explicit, test_c_implicit, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, args):
    mapping_test_explicit = d_f(test_c_explicit)
    residual_test_implicit = test_c_implicit - mapping_test_explicit
    if args.dataset == 'CUB':
        cut = torch.LongTensor([[1] * 11 + [0] * (args.n_attributes-11)] * test_c_explicit.shape[0])
    else:
        # cut = torch.LongTensor([[1] * int(args.n_attributes * 0.2) + [0] * (args.n_attributes-int(args.n_attributes * 0.2))] * test_c_explicit.shape[0])
        cut = torch.LongTensor([[0,1,1,1,0,0,0,0]] * test_c_explicit.shape[0])
    cut = cut.to(args.device)
    test_c_explicit_new = (torch.sign(test_c_explicit*cut)==torch.sign(test_c_truth.to(args.device)*cut)) * test_c_explicit + \
                (torch.sign(test_c_explicit*cut)>torch.sign(test_c_truth.to(args.device)*cut)) * quantile05 + \
                (torch.sign(test_c_explicit*cut)<torch.sign(test_c_truth.to(args.device)*cut)) * quantile95
    test_y_new = torch.mm(test_c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + torch.mm(test_c_implicit, W_implicit.t())
    acc_origin = ((test_y_implicit + test_y_explicit).max(axis=1)[1] == test_y_truth.to(args.device)).sum()/test_y_truth.shape[0]
    acc_int = (test_y_new.max(axis=1)[1] == test_y_truth.to(args.device)).sum()/test_y_truth.shape[0]
    test_c_implicit_new = d_f(test_c_explicit_new).detach() + residual_test_implicit
    
    mine_y_new = torch.mm(test_c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + torch.mm(test_c_implicit_new, W_implicit.t())
    acc_mine = ((mine_y_new).max(axis=1)[1] == test_y_truth.to(args.device)).sum()/test_y_truth.shape[0]
    return acc_origin, acc_int, acc_mine
    
def new_loss(train_c_explicit, train_c_implicit, train_y_truth, train_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, criterion, args):
    mapping_train_explicit = d_f(train_c_explicit)
    residual_train_implicit = train_c_implicit - mapping_train_explicit
    
    #train_c_explicit.shape=(-1, 112)
    if args.dataset == 'CUB':
        cut = torch.LongTensor([[1] * 11 + [0] * (args.n_attributes-11)] * train_c_explicit.shape[0])
    else:
        # cut = torch.LongTensor([[1] * int(args.n_attributes * 0.5) + [0] * (args.n_attributes-int(args.n_attributes * 0.5))] * train_c_explicit.shape[0])
        cut = torch.LongTensor([[0,1,1,1,0,0,0,0]] * train_c_explicit.shape[0])
    cut = cut.to(args.device)
    train_c_explicit_new = (torch.sign(train_c_explicit*cut)==torch.sign(train_c_truth.to(args.device)*cut)) * train_c_explicit + \
                (torch.sign(train_c_explicit*cut)>torch.sign(train_c_truth.to(args.device)*cut)) * quantile05 + \
                (torch.sign(train_c_explicit*cut)<torch.sign(train_c_truth.to(args.device)*cut)) * quantile95
    train_y_new = torch.mm(train_c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + torch.mm(train_c_implicit, W_implicit.t())
    train_c_implicit_new = d_f(train_c_explicit_new).detach() + residual_train_implicit
    mine_y_new = torch.mm(train_c_explicit_new, W_explicit.t()) + b_explicit + b_implicit + torch.mm(train_c_implicit_new, W_implicit.t())
    loss = criterion(mine_y_new.to(args.device), train_y_truth.to(args.device))
    return loss

if __name__ == '__main__':
    args = parse_arguments()

    cuda_name = "cuda:0"
    device = torch.device("cuda:" + str(args.cuda_device) if args.use_cuda else "cpu")
    args.device = device

    
    if args.inference:
        
        logger = Logger(os.path.join(args.log_dir, 'log_mine_inference.txt'))
        logger.write(str(args) + '\n')
        logger.flush()
        test_data_dir = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
        test_loader = load_data([test_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                   n_class_attr=args.n_class_attr, concept_percent = args.concept_percent, dataset=args.dataset)
        
        k = 0
        acc_origin_list, acc_int_list, acc_mine_list = [], [], []
        time_list = []
        for logs in args.log_dir2:
            t0 = time()
            model = torch.load(args.model_dir2[k], map_location=cuda_name)
            k += 1
            model.eval()
            W_explicit = model.sub_models.sub_model1.linear.weight.detach()
            b_explicit = model.sub_models.sub_model1.linear.bias.detach()
            W_implicit = model.sub_models.sub_model2.linear.weight.detach()
            b_implicit = model.sub_models.sub_model2.linear.bias.detach()
            test_c_explicit, test_c_implicit, test_y, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth = data_gradients(test_loader, model)
            quantile95 = torch.load(os.path.join(logs, 'quantile95'), map_location=cuda_name)
            quantile05 = torch.load(os.path.join(logs, 'quantile05'), map_location=cuda_name)
            d_f = torch.load(os.path.join(logs, 'mine.pth'), map_location=cuda_name)
            acc_origin, acc_int, acc_mine = acc(test_c_explicit, test_c_implicit, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, d_f, quantile95, quantile05, args)
            logger.write('acc_origin: %.4f, acc_int: %.4f, acc_mine: %.4f\n' % (acc_origin, acc_int, acc_mine))
            acc_origin_list.append(acc_origin.item())
            acc_int_list.append(acc_int.item())
            acc_mine_list.append(acc_mine.item())
            t1 = time()
            time_list.append(t1-t0)
            logger.write('time: %ds\n' % (t1-t0))
        logger.write('acc_origin: %.4f+-%.4f, acc_int: %.4f+-%.4f, acc_mine: %.4f+-%.4f, time: %.4f+-%.4f\n' % (np.mean(acc_origin_list), np.std(acc_origin_list), np.mean(acc_int_list), np.std(acc_int_list), np.mean(acc_mine_list), np.std(acc_mine_list), np.mean(time_list), np.std(time_list)))


    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = torch.load(args.model_dir, map_location=cuda_name)
        model.eval()   
        W_explicit = model.sub_models.sub_model1.linear.weight.detach()
        b_explicit = model.sub_models.sub_model1.linear.bias.detach()
        W_implicit = model.sub_models.sub_model2.linear.weight.detach()
        b_implicit = model.sub_models.sub_model2.linear.bias.detach()
        
        train_data_dir = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        val_data_dir = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
        test_data_dir = os.path.join(BASE_DIR, args.data_dir, 'test.pkl')
        train_loader = load_data([train_data_dir, val_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, concept_percent = args.concept_percent, dataset=args.dataset, training_opt=False)
        
        # test_loader = load_data([test_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, concept_percent = args.concept_percent, dataset=args.dataset, training_opt=False)
                            
        logger = Logger(os.path.join(args.log_dir, 'log_mine.txt'))
        logger.write(str(args) + '\n')
        logger.flush()
        
        # test_c_explicit, test_c_implicit, test_y, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth = data_gradients(test_loader, model)
        
        train_c_explicit, train_c_implicit, train_y, train_y_explicit, train_y_implicit, train_y_truth, train_c_truth = data_gradients(train_loader, model)
        quantile95 = torch.quantile(train_c_explicit, 0.95, dim=0)
        quantile05 = torch.quantile(train_c_explicit, 0.05, dim=0)
        torch.save(quantile95, os.path.join(args.log_dir, 'quantile95'))
        torch.save(quantile05, os.path.join(args.log_dir, 'quantile05'))
        t0 = time()

        d_f = run_mine(args, train_c_explicit, train_c_implicit, train_y_explicit, train_y_implicit, train_y_truth, train_c_truth, W_explicit, b_explicit, W_implicit, b_implicit, quantile95, quantile05
        # , test_c_explicit, test_c_implicit, test_y_truth, test_c_truth
        )
        t1 = time()
        logger.write('time: %ds\n' % (t1-t0))
        

