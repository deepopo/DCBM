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
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import math
from time import time
import pickle as pkl

import argparse
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
    return c_explicit, c_implicit, y, y_explicit, y_implicit, y_truth, c_truth

if __name__ == '__main__':
    args = parse_arguments()

    cuda_name = "cuda:0"
    device = torch.device("cuda:" + str(args.cuda_device) if args.use_cuda else "cpu")
    args.device = device
    
    criterion = torch.nn.CrossEntropyLoss()


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
    train_loader = load_data([train_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, concept_percent = args.concept_percent, dataset=args.dataset, training_opt=False)
    
    test_loader = load_data([test_data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, concept_percent = args.concept_percent, dataset=args.dataset, training_opt=False)
                        
    logger = Logger(os.path.join(args.log_dir, 'log_rec.txt'))
    logger.write(str(args) + '\n')
    logger.flush()

    
    test_c_explicit, test_c_implicit, test_y, test_y_explicit, test_y_implicit, test_y_truth, test_c_truth = data_gradients(test_loader, model)
    ### TEST
    
    d_f = torch.load(os.path.join(args.log_dir, 'mine.pth'), map_location=cuda_name)

    acc_list = []
    all_origin_list = []
    all_origin_to01_list = []
    all_predict_list = []
    all_predict_to01_list = []
    all_predict2_list = []
    all_predict2_to01_list = []
    all_truth_list = []
    t = 0
    num = 0
    for anchor in range(len(test_y)):
        c0 = test_c_explicit.clone().detach()
        c_explicit_var = torch.autograd.Variable(c0.clone().detach()).to(args.device)
        c_explicit_var.requires_grad = True
        # optimizer = torch.optim.SGD([c_explicit_var], lr=0.025, momentum=0.9, weight_decay=0)
        optimizer = torch.optim.Adam([c_explicit_var], lr=0.025, weight_decay=0)        
        alpha = 0

        if accuracy(test_y[anchor].unsqueeze(0), test_y_truth[anchor].unsqueeze(0), args, topk=[1])[0][0].data.cpu().numpy() != 0:
            continue
            
        test_c_sigmoid = torch.nn.Sigmoid()(c0)
        acc = binary_accuracy(test_c_sigmoid[anchor, :], test_c_truth[anchor, :])
        origin_acc = acc.data.cpu().numpy()
        
        all_truth_list.extend(list(test_c_truth[anchor, :].data.cpu().numpy()))
        all_origin_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        test_c_sigmoid = torch.sign(test_c_sigmoid - 0.5)
        test_c_sigmoid = (test_c_sigmoid + 1.0) / 2
        all_origin_to01_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        for step in range(args.epochs):
            test_c_sigmoid = torch.nn.Sigmoid()(c0)
            if step != 0:
                pre_loss = loss.item()
            l1 = criterion((c_explicit_var[anchor].matmul(W_explicit.T) + b_explicit + test_y_implicit[anchor]).unsqueeze(0), test_y_truth[anchor].unsqueeze(0).to(args.device))

            l2 = alpha * torch.norm(c_explicit_var[anchor] - c0[anchor]) ** 2
            loss = l1 + l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step > 10) and (abs(pre_loss - loss.item()) <= 10e-5):
                break

        test_c_sigmoid = torch.nn.Sigmoid()(c_explicit_var)
        acc = binary_accuracy(test_c_sigmoid[anchor, :], test_c_truth[anchor, :])
        acc = acc.data.cpu().numpy()

        all_predict_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        test_c_sigmoid = torch.sign(test_c_sigmoid - 0.5)
        test_c_sigmoid = (test_c_sigmoid + 1.0) / 2
        all_predict_to01_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        # with decouple
        num += 1
        t0 = time()
        
        c0 = test_c_explicit.clone().detach()
        c_explicit_var2 = torch.autograd.Variable(c0.clone().detach()).to(args.device)
        c_implicit_var2 = torch.autograd.Variable(test_c_implicit.clone().detach()).to(args.device)
        c_explicit_var2.requires_grad = True
        
        # optimizer2 = torch.optim.SGD([c_explicit_var2], lr=0.025, momentum=0.9, weight_decay=0)
        optimizer2 = torch.optim.Adam([c_explicit_var2], lr=0.025, weight_decay=0)
        mapping_test_explicit = d_f(c_explicit_var2)
        residual_test_implicit = (c_implicit_var2 - mapping_test_explicit).detach()

        for step in range(args.epochs):
            test_c_sigmoid = torch.nn.Sigmoid()(c0)
            if step != 0:
                pre_loss2 = loss2.item()
            mapping_test_explicit = d_f(c_explicit_var2)
            c_implicit_new = mapping_test_explicit + residual_test_implicit
            l3 = criterion((c_explicit_var2[anchor].matmul(W_explicit.T) + b_explicit + b_implicit + c_implicit_new[anchor].matmul(W_implicit.T)).unsqueeze(0), test_y_truth[anchor].unsqueeze(0).to(args.device))

            l4 = alpha * torch.norm(c_explicit_var2[anchor] - c0[anchor]) ** 2
            loss2 = l3 + l4
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            if (step > 10) and (abs(pre_loss2 - loss2.item()) <= 10e-5):
                break

        test_c_sigmoid = torch.nn.Sigmoid()(c_explicit_var2)
        acc2 = binary_accuracy(test_c_sigmoid[anchor, :], test_c_truth[anchor, :])
        acc2 = acc2.data.cpu().numpy()
        all_predict2_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        # origin_auc = roc_auc_score(test_c_truth[anchor, :].data.cpu().numpy(), test_c_sigmoid[anchor, :].data.cpu().numpy())
        test_c_sigmoid = torch.sign(test_c_sigmoid - 0.5)
        test_c_sigmoid = (test_c_sigmoid + 1.0) / 2
        all_predict2_to01_list.extend(list(test_c_sigmoid[anchor, :].data.cpu().numpy()))
        
        logger.write('Anchor: %d, origin acc: %.4f, rec acc: %.4f, dec acc: %.4f\n' % (anchor, origin_acc, acc, acc2))
        logger.flush()
        acc_list.append([origin_acc, acc, acc2])
        
        t1 = time()
        t += (t1 - t0)
        
        
    acc_list = np.array(acc_list)
    acc_mean = acc_list.mean(axis=0)
    origin_auc = roc_auc_score(all_truth_list, all_origin_list)
    auc = roc_auc_score(all_truth_list, all_predict_list)
    auc2 = roc_auc_score(all_truth_list, all_predict2_list)
    
    predict_explicitrong_num = np.sum((np.array(all_truth_list) != np.array(all_origin_to01_list)))
    rec_correct = np.sum((np.array(all_predict_to01_list) != np.array(all_origin_to01_list)) * (np.array(all_truth_list) != np.array(all_origin_to01_list)) )
    rec_explicitrong = np.sum((np.array(all_predict_to01_list) != np.array(all_origin_to01_list)) * (np.array(all_truth_list) == np.array(all_origin_to01_list)) )
    dec_correct = np.sum((np.array(all_predict2_to01_list) != np.array(all_origin_to01_list)) * (np.array(all_truth_list) != np.array(all_origin_to01_list)) )
    dec_explicitrong = np.sum((np.array(all_predict2_to01_list) != np.array(all_origin_to01_list)) * (np.array(all_truth_list) == np.array(all_origin_to01_list)) )
    
    origin_fpr, origin_tpr, origin_thresholds = roc_curve(all_truth_list, all_origin_to01_list, pos_label=1)
    origin_fpr = origin_fpr[1]
    origin_tpr = origin_tpr[1]
    fpr, tpr, thresholds = roc_curve(all_truth_list, all_predict_to01_list, pos_label=1)
    fpr = fpr[1]
    tpr = tpr[1]
    fpr2, tpr2, thresholds2 = roc_curve(all_truth_list, all_predict2_to01_list, pos_label=1)
    fpr2 = fpr2[1]
    tpr2 = tpr2[1]
    
    logger.write('origin acc: %.4f, rec acc: %.4f, dec acc: %.4f\norigin AUC: %.4f, rec AUC: %.4f, dec AUC: %.4f\norigin FPR: %.4f, rec FPR: %.4f, dec FPR: %.4f\norigin TPR: %.4f, rec TPR: %.4f, dec TPR: %.4f\n' % (acc_mean[0], acc_mean[1], acc_mean[2], origin_auc, auc, auc2, origin_fpr, fpr, fpr2, origin_tpr, tpr, tpr2))
    logger.write('total num: %d, predict wrong num: %d, rec correct: %d, rec wrong %d, dec correct: %d, dec wrong %d\n' % (len(all_truth_list), predict_explicitrong_num, rec_correct, rec_explicitrong, dec_correct, dec_explicitrong))
    logger.write('time: %.4fs\n' % (t/num))
    

