# -*- coding: utf-8 -*-
import os
import glob
import torch
import argparse
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models import DCBMInterface, MINEInterface, RECInterface
from data import DInterface, CelebAInterface, MINEDataInterface
from utils import data_gradients

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default="CUB_int", help='config file')
    parser.add_argument('-seed', type=int, help='seed.')
    return parser.parse_args()

def train(args):
    config = OmegaConf.load(f"configs/{args.d}.yaml")
    seed = args.seed if args.seed else config.base.seed
    pl.seed_everything(seed)

    # copy parameters
    mode = config.base.mode
    max_epochs = config.base.max_epochs
    dataset = config.base.dataset
    data_dir = config.base.data_dir
    log_dir = config.base.log_dir
    save_dir = config.base.save_dir
    ckpt_path = None if config.base.ckpt_path == 'None' else config.base.ckpt_path
    model_name = config.base.model_name
    batch_size = config.base.batch_size

    backbone = config.base.backbone
    n_attributes = config.base.n_attributes
    num_classes = config.base.num_classes
    use_attr = config.base.use_attr
    no_img = config.base.no_img
    concept_percent = config.base.concept_percent
    n_class_attr = config.base.n_class_attr

    config.data.params.n_attributes = n_attributes
    config.data.params.use_attr = use_attr
    config.data.params.no_img = no_img
    config.data.params.concept_percent = concept_percent
    config.data.params.n_class_attr = n_class_attr

    config.model.concept_percent = concept_percent
    config.model.save_dir = save_dir
    config.model.dcbm_config.params.n_attributes = n_attributes
    config.model.dcbm_config.params.num_classes = num_classes
    config.model.dcbm_config.params.backbone = backbone
    config.model.dcbm_config.params.n_class_attr = n_class_attr

    # define data and model interfaces.
    if dataset == 'celeba':
        data_module = CelebAInterface(config.data, data_dir, save_dir, batch_size)
    else:
        data_module = DInterface(config.data, data_dir, batch_size)
    model = DCBMInterface(  config.model, 
                            use_attr, 
                            no_img, 
                            data_dir, 
                            n_attributes, 
                            batch_size, 
                            max_epochs, 
                            dataset)
    
    # define logger
    log_name = f'{model_name}_{dataset}_{backbone}_cp{concept_percent}_seed{seed}.log'
    root_log_dir = os.path.join(log_dir, log_name)
    mine_log_name = f'{model_name}_{dataset}_{backbone}_cp{concept_percent}_seed{seed}_int.log'
    mine_root_log_dir = os.path.join(log_dir, mine_log_name)

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        version_dirs = sorted(glob.glob(os.path.join(root_log_dir, "version_*")), key=os.path.getmtime)
        latest_version_dir = version_dirs[-1] if version_dirs else None
        checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
        best_checkpoint = glob.glob(f"{checkpoint_dir}/best-epoch=*.ckpt")
        if best_checkpoint:
            best_checkpoint_path = best_checkpoint[0]
            print(f"Best checkpoint path: {best_checkpoint_path}")
            model = DCBMInterface.load_from_checkpoint(best_checkpoint_path)
        else:
            print("No best checkpoint found in the latest version.")

    model.eval()
    W_explicit = model.dcbm.model.sub_models.sub_model1.linear.weight.detach()
    b_explicit = model.dcbm.model.sub_models.sub_model1.linear.bias.detach()
    W_implicit = model.dcbm.model.sub_models.sub_model2.linear.weight.detach()
    b_implicit = model.dcbm.model.sub_models.sub_model2.linear.bias.detach()

    if mode in ['test', 'both']:
        test_loader = data_module.test_dataloader()
        test_c_explicit, test_c_implicit, test_c_truth, test_y_explicit, test_y_implicit, test_y_truth = data_gradients(test_loader, model, dataset)
        test_data = [test_c_explicit, test_c_implicit, test_c_truth, test_y_explicit, test_y_implicit, test_y_truth, W_explicit, b_explicit, W_implicit, b_implicit]
        dim_explicit = test_c_explicit.shape[1]
        dim_implicit = test_c_implicit.shape[1]
    mine_data_module = MINEDataInterface(train_data=test_data, test_data=test_data)

    config.int_model.decouple_config.params.dim_explicit = dim_explicit
    config.int_model.decouple_config.params.dim_implicit = dim_implicit
    config.int_model.mine_config.params.dim_explicit = dim_explicit
    config.int_model.mine_config.params.dim_implicit = dim_implicit
    mine_steps = config.int_model.mine_steps
    eta = config.int_model.eta
    mine_model = MINEInterface(config.int_model, 
                               mine_steps, 
                               dataset, 
                               n_attributes, 
                               eta)

    # choose to load checkpoints
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        version_dirs = sorted(glob.glob(os.path.join(mine_root_log_dir, "version_*")), key=os.path.getmtime)
        latest_version_dir = version_dirs[-1] if version_dirs else None
        checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
        last_checkpoint = glob.glob(f"{checkpoint_dir}/epoch=*.ckpt")
        if last_checkpoint:
            checkpoint_path = last_checkpoint[0]
            print(f"checkpoint path: {checkpoint_path}")
            mine_model = MINEInterface.load_from_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found in the latest version.")

    rec_steps = config.int_model.rec_steps
    alpha = config.int_model.alpha
    rec_lr = config.int_model.rec_lr
    rec = RECInterface(mine_model.d_f, alpha, rec_lr, rec_steps)
    trainer = Trainer(accelerator="gpu", max_epochs=1, callbacks=[], logger=False, enable_checkpointing=False) # don't save checkpoints
    trainer.fit(rec, mine_data_module)
    trainer.test(rec, mine_data_module)

if __name__ == '__main__':
    args = get_args()
    train(args)