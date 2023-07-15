import argparse
from utils.config import N_ATTRIBUTES

def parse_arguments():
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('-dataset', type=str, help='Name of the dataset.')
    parser.add_argument('-exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch', 'DCBM'],
                        default='DCBM',
                        help='Name of experiment to run.')
    parser.add_argument('-seed', default=1, type=int, help='Numpy and torch seed.')
    parser.add_argument('-log_dir', default="./logs/", help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, default=64, help='mini-batch size')
    parser.add_argument('-num_classes', type=int, default = 200, help='dimension of hidden_layer')
    parser.add_argument('-epochs', '-e', type=int, default = 1000, help='epochs for training process')
    parser.add_argument('-implicit_dim', type=int, default = 256, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=4e-4, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_cuda', default=True, type=bool, help='where to use cuda')
    parser.add_argument('-cuda_device', default=0, type=int, help='which cuda device to use')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
    parser.add_argument('-embs_weight', default=0.001, type=float, help='weight of JS Divergence')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-use_embs', help='whether to use embedding penalty when training', action='store_true')
    parser.add_argument('-weighted_loss', default='multiple', # note: may need to reduce lr
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-uncertain_labels', action='store_true',
                        help='whether to use (normalized) attribute certainties as labels')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-expand_dim', type=int, default=0,
                        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default="CUB_processed/class_attr_data_10", help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', type=bool, default=False, help='For retraining on both train + val set')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                             'For end2end & bottleneck model')
    parser.add_argument('-early_stop', action='store_true',
                        help='Whether to perform early stopping while training. ')
    parser.add_argument('-connect_CY', action='store_true',
                        help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
    parser.add_argument('-concept_percent', type=int,
                        choices=[100, 50, 40, 30, 20, 10],
                        default=100,
                        help='Use how many concepts (Ratio).')
    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    return args
