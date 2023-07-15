import sys
from utils.args import parse_arguments
from utils.template_model import inception_v3, MLP, ConvergeEnd2EndModel, DisperseEnd2EndModel
from utils.config import BASE_DIR, N_CLASSES, MIN_LR, LR_DECAY_SIZE
from utils.dataset import load_data, find_class_imbalance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
from utils.analysis import Logger, AverageMeter
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.run import run_epoch
from time import time
import numpy as np
args = parse_arguments()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if not torch.cuda.is_available():
    args.use_cuda = False

if not args.use_embs:
    args.embs_weight = 0

if not args.use_attr:
    args.attr_loss_weight = 0

### For example, DCBM0.001_0.01_100Model_Seed1
args.log_dir = os.path.join(args.log_dir, args.exp + str(args.attr_loss_weight) + \
                                str('_') + str(args.embs_weight) + \
                                str('_') + str(args.concept_percent) + \
                                "Model_Seed" + str(args.seed))

device = torch.device("cuda:" + str(args.cuda_device) if args.use_cuda else "cpu")
args.device = device

### model structure
CNN_Layer = inception_v3(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.num_classes, aux_logits=args.use_aux,
                          n_attributes=args.n_attributes+args.implicit_dim, bottleneck=True, expand_dim=args.expand_dim,
                          three_class=(args.n_class_attr == 3))
Explicit_MLP_Layer = MLP(input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim)
Implicit_MLP_Layer = MLP(input_dim=args.implicit_dim, num_classes=args.num_classes, expand_dim=args.expand_dim)
Embedding_Layer = DisperseEnd2EndModel(CNN_Layer, Explicit_MLP_Layer, Implicit_MLP_Layer, 0, args.n_attributes, args.n_attributes, args.n_attributes+args.implicit_dim, args.use_relu, args.use_sigmoid)

### the outputs of Converge_MLP_Layer are not used in the paper.
Converge_MLP_Layer = MLP(input_dim=args.num_classes*2, num_classes=args.num_classes, expand_dim=args.expand_dim)
model = ConvergeEnd2EndModel(Embedding_Layer, Converge_MLP_Layer, 0, 0, args.use_relu, args.use_sigmoid)

###
train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
imbalance = None
if args.use_attr and not args.no_img and args.weighted_loss:
    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    if args.weighted_loss == 'multiple':
        imbalance = find_class_imbalance(train_data_path, True, concept_percent=args.concept_percent)
    else:
        imbalance = find_class_imbalance(train_data_path, False, concept_percent=args.concept_percent)
model = model.to(device)
if os.path.exists(args.log_dir): # job restarted by cluster
    for f in os.listdir(args.log_dir):
        if 'bak' not in f:
            os.remove(os.path.join(args.log_dir, f))
else:
    os.makedirs(args.log_dir)
logger = Logger(os.path.join(args.log_dir, 'log.txt'))
logger.write(str(args) + '\n')
logger.write(str(imbalance) + '\n')
logger.flush()

criterion = torch.nn.CrossEntropyLoss()
attr_criterion = [] #separate criterion (loss function) for each attribute
if args.weighted_loss:
    assert(imbalance is not None)
    for ratio in imbalance:
        attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio])).to(device))
else:
    for i in range(args.n_attributes):
        attr_criterion.append(torch.nn.CrossEntropyLoss())
import math
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
print("Stop epoch: ", stop_epoch)

train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
logger.write('train data path: %s\n' % train_data_path)
logger.write('validation data path: %s\n' % val_data_path)

if args.ckpt: #retraining
    train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                             n_class_attr=args.n_class_attr, resampling=args.resampling, concept_percent=args.concept_percent, dataset=args.dataset)
    val_loader = None
else:
    train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                             n_class_attr=args.n_class_attr, resampling=args.resampling, concept_percent=args.concept_percent, dataset=args.dataset)
    val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, concept_percent=args.concept_percent, dataset=args.dataset)

best_val_epoch = -1
best_val_loss = float('inf')
best_val_acc = 0
t0 = time()
for epoch in range(0, args.epochs):
    t1 = time()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    if args.use_embs:
        train_loss_meter, train_acc_meter, train_acc_attr_meter, loss_meter_main, loss_meter_concept, loss_meter_penalty, loss_meter_implicit, loss_meter_explicit = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
    else:
        train_loss_meter, train_acc_meter, train_acc_attr_meter, loss_meter_main, loss_meter_concept = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
        loss_meter_penalty, loss_meter_implicit, loss_meter_explicit = 0, 0, 0

    if not args.ckpt: # evaluate on val set
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        with torch.no_grad():
            val_loss_meter, val_acc_meter, val_acc_attr_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

    else: #retraining
        val_loss_meter = train_loss_meter
        val_acc_meter = train_acc_meter
        val_acc_attr_meter = train_acc_attr_meter

    if best_val_acc < val_acc_meter.avg:
        best_val_epoch = epoch
        best_val_acc = val_acc_meter.avg
        logger.write('New model best model at epoch %d\n' % epoch)
        torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
        #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
        #    break
    t2 = time()
    if args.use_embs:
        logger.write('Epoch [%d]:\tTrain: (loss): %.4f\t(acc): %.4f\t(attr acc): %.4f\n'
                'Val: (loss): %.4f\t(acc): %.4f\t(attr acc): %.4f\n'
                'Train loss: (main): %.4f\t(concept): %.4f\t(penalty): %.4f\t(implicit): %.4f\t(explicit): %.4f\n'
                'Best val epoch: %d\n'
                'Time for the current iteration: %.2fs\tTotal time: %.2fs\n'
                % (epoch, train_loss_meter.avg, train_acc_meter.avg, train_acc_attr_meter.avg, \
                   val_loss_meter.avg, val_acc_meter.avg, val_acc_attr_meter.avg, \
                   loss_meter_main.avg, loss_meter_concept.avg, loss_meter_penalty.avg, loss_meter_implicit.avg, loss_meter_explicit.avg, \
                   best_val_epoch, \
                   t2 - t1, t2 - t0))
    else:
        logger.write('Epoch [%d]:\tTrain: (loss): %.4f\t(acc): %.4f\t(attr acc): %.4f\n'
                'Val: (loss): %.4f\t(acc): %.4f\t(attr acc): %.4f\n'
                'Train loss: (main): %.4f\t(concept): %.4f\tBest val epoch: %d\n'
                'Time for the current iteration: %.2fs\tTotal time: %.2fs\n'
                % (epoch, train_loss_meter.avg, train_acc_meter.avg, train_acc_attr_meter.avg, \
                   val_loss_meter.avg, val_acc_meter.avg, val_acc_attr_meter.avg, \
                   loss_meter_main.avg, loss_meter_concept.avg, best_val_epoch, \
                   t2 - t1, t2 - t0))
    logger.flush()

    if epoch <= stop_epoch:
        scheduler.step(epoch) #scheduler step to update lr at the end of epoch
    #inspect lr
    if epoch % 10 == 0:
        print('Current lr:', scheduler.get_last_lr())

    # if epoch % args.save_step == 0:
    #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))
    if args.early_stop:
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

# torch.save(model, os.path.join(args.log_dir, 'last_model_%d.pth' % args.seed))




