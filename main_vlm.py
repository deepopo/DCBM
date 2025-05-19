import torch
import os
import random
import utils
import argparse
import datetime
import json

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from utils import cos_similarity_cubed_single, zero_out_small_weights
from utils import LABEL_FILES, save_activations, get_save_names, get_targets_only
from utils.analysis import js_div
from models import threshold_linear

parser = argparse.ArgumentParser(description='Settings for creating DCBM')

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saves', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='logs', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.0, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=5000, help="how many steps to train the projection layer")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")

parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight decay for optimizer')
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--l1_lambda", type=float, default=0.0001, help="weight of l1 norm")
parser.add_argument("--threshold", type=float, default=0.0001, help="threshold to clip the small values")

def train_cbm_and_save(args):
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "dataset/concept_sets/{}_filtered.txt".format(args.dataset)
        
    criterion = torch.nn.CrossEntropyLoss()
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    l1_lambda = args.l1_lambda
    
    #get concept set
    cls_file = LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")
    
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                         target_layers = [args.feature_layer], d_probe = d_probe,
                         concept_set = args.concept_set, batch_size = args.batch_size, 
                         device = args.device, pool_mode = "avg", save_dir = args.activation_dir)
        
    target_save_name, clip_save_name, text_save_name = get_save_names(args.clip_name, args.backbone, 
                                            args.feature_layer,d_train, args.concept_set, "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(args.clip_name, args.backbone,
                                            args.feature_layer, d_val, args.concept_set, "avg", args.activation_dir)
    
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features
    
    #filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)
    
    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i]<=args.clip_cutoff:
                print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))


    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.clip_cutoff]

    #save memory by recalculating 
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>args.clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
    
        clip_features = image_features @ text_features.T
        del image_features, text_features
    
    val_clip_features = val_clip_features[:, highest>args.clip_cutoff]

    train_targets = get_targets_only(d_train)
    val_targets = get_targets_only(d_val)
    
    with torch.no_grad():
        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)
    labels_var = torch.autograd.Variable(train_y)
    labels_var = labels_var.to(args.device)
    val_labels_var = torch.autograd.Variable(val_y)
    val_labels_var = val_labels_var.to(args.device)

    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts)+128,
                                 bias=False).to(args.device)
    threshold = args.threshold
    linear_exp = threshold_linear(len(concepts), len(classes), threshold).to(args.device)
    linear_imp = torch.nn.Linear(128, len(classes)).to(args.device)

    params = list(proj_layer.parameters()) + list(linear_exp.parameters()) + list(linear_imp.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=args.weight_decay)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    alpha = 0.1
    beta = 1.0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    out_dict = {}
    out_dict['metrics'] = {}
    out_dict['alpha'] = alpha
    out_dict['beta'] = beta
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs_c = proj_layer(target_features[batch].to(args.device).detach())
        labels = labels_var[batch]
        outs_c_explicit = outs_c[:, :len(concepts)]
        outs_c_implicit = outs_c[:, len(concepts):]

        loss_c = -cos_similarity_cubed_single(clip_features[batch].to(args.device).detach(), outs_c_explicit)
        loss_c = alpha * torch.sum(loss_c) / (1 + alpha * len(concepts))
        outs_y_explicit = linear_exp(outs_c_explicit)
        outs_y_implicit = linear_imp(outs_c_implicit)

        loss_y = criterion(outs_y_explicit + outs_y_implicit, labels)
        loss_implicit = js_div(outs_y_explicit, outs_y_explicit + outs_y_implicit)
        loss_penalty = beta * loss_implicit
        l1_norm = sum(((torch.sign(p.abs()-threshold))*p.abs()).sum() for name, p in linear_exp.named_parameters() if "weight" in name)
        loss = loss_y + loss_c + loss_penalty + l1_lambda*l1_norm

        loss.backward()
        opt.step()
        out_dict['threshold'] = threshold
        out_dict['l1_lambda'] = l1_lambda
        zero_out_small_weights(linear_exp, threshold)
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_labels = val_labels_var
                val_outs_c_explicit = val_output[:, :len(concepts)]
                val_outs_c_implicit = val_output[:, len(concepts):]

                val_loss_c = -cos_similarity_cubed_single(val_clip_features.to(args.device).detach(), val_outs_c_explicit)
                val_loss_c = alpha * torch.sum(val_loss_c) / (1 + alpha * len(concepts))

                val_outs_y_explicit = linear_exp(val_outs_c_explicit)
                val_outs_y_implicit = linear_imp(val_outs_c_implicit)

                val_loss_y = criterion(val_outs_y_explicit + val_outs_y_implicit, val_labels)
                val_loss_implicit = js_div(val_outs_y_explicit, val_outs_y_explicit + val_outs_y_implicit)
                val_loss_penalty = beta * val_loss_implicit

                val_loss = val_loss_y + val_loss_c + val_loss_penalty
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                print('train loss: y: {:.4f}, c: {:.4f}, js: {:.4f}'.format(loss_y, loss_c, loss_penalty))
                print('val loss: y: {:.4f}, c: {:.4f}, js: {:.4f}'.format(val_loss_y, val_loss_c, val_loss_penalty))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                out_dict['metrics']['loss_y'] = loss_y.cpu().item()
                out_dict['metrics']['loss_c'] = loss_c.cpu().item()
                out_dict['metrics']['loss_penalty'] = loss_penalty.cpu().item()
                out_dict['metrics']['val_loss_y'] = val_loss_y.cpu().item()
                out_dict['metrics']['val_loss_c'] = val_loss_c.cpu().item()
                out_dict['metrics']['val_loss_penalty'] = val_loss_penalty.cpu().item()

                _, pred = (val_outs_y_explicit + val_outs_y_implicit).topk(1, 1, True, True)
                pred = pred.t()
                temp = val_labels.view(1, -1).expand_as(pred)
                temp = temp.to(args.device)
                correct = pred.eq(temp)
                correct = correct[:1].reshape(-1).float().mean(0, keepdim=True) * 100
                out_dict['metrics']['acc'] = correct.item()
                _, pred = (val_outs_y_explicit).topk(1, 1, True, True)
                pred = pred.t()
                temp = val_labels.view(1, -1).expand_as(pred)
                temp = temp.to(args.device)
                exp_correct = pred.eq(temp)
                exp_correct = exp_correct[:1].reshape(-1).float().mean(0, keepdim=True) * 100
                out_dict['metrics']['exp_acc'] = exp_correct.item()
                W_g = linear_exp.weight * (torch.abs(linear_exp.weight)>threshold)
                nnz = (W_g.abs() > 1e-5).sum().item()
                total = W_g.numel()
                sparsity = nnz/total
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss_c.cpu(),
                                                                                               -val_loss_c.cpu()))
                print('train loss: y: {:.4f}, c: {:.4f}, js: {:.4f}'.format(loss_y.cpu(), loss_c.cpu(), loss_penalty.cpu()))
                print('val loss: y: {:.4f}, c: {:.4f}, js: {:.4f}'.format(val_loss_y.cpu(), val_loss_c.cpu(), val_loss_penalty.cpu()))
                print('acc: {:.4f}, exp_acc: {:.4f}, sparsity: {:.4f}'.format(correct.item(), exp_correct.item(), sparsity))
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        
    W_g = linear_exp.weight * (torch.abs(linear_exp.weight)>threshold)

    save_name = "{}/{}_cbm_{}_{}".format(args.save_dir, args.dataset, args.seed, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        out_dict['seed'] = args.seed
        json.dump(out_dict, f, indent=2)
    print('sparsity: {:.6f}'.format(nnz/total))
    
if __name__=='__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    train_cbm_and_save(args)