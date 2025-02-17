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

from models import DCBMInterface
from data import DInterface, CelebAInterface
from utils import load_callbacks, examine_dir

torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default="CUB", help='config file')
    parser.add_argument('-seed', type=int, help='seed.')
    return parser.parse_args()

def train(args):
    config = OmegaConf.load(f"configs/{args.d}.yaml")
    seed = args.seed if args.seed else config.base.seed
    pl.seed_everything(seed)

    # copy parameters
    mode = config.base.mode
    max_epochs = config.base.max_epochs
    patience = config.base.patience
    data_dir = config.base.data_dir
    batch_size = config.base.batch_size
    log_dir = config.base.log_dir
    save_dir = config.base.save_dir
    ckpt_path = None if config.base.ckpt_path == 'None' else config.base.ckpt_path
    model_name = config.base.model_name
    dataset = config.base.dataset
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

    # generate log_dir if not existed
    examine_dir(log_dir)
    # generate save_dir if not existed
    examine_dir(save_dir)

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

    if dataset in ['celeba']:
        callbacks = load_callbacks(monitor='val_acc_label', patience=patience, mode='max')
    else:
        # the vanilla CBM combines and retrains train/val datasets after selecting best hyperparameters.
        callbacks = load_callbacks(monitor='train_acc_label', patience=patience, mode='max')

    # define logger
    log_name = f'{model_name}_{dataset}_{backbone}_cp{concept_percent}_seed{seed}.log'
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name)
    root_log_dir = os.path.join(log_dir, log_name)

    # define trainner
    trainer = Trainer(accelerator="gpu", 
                      max_epochs=max_epochs, 
                      callbacks=callbacks, 
                      logger=logger)
    
    # choose to load checkpoints
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    # choose train/test
    if mode in ['train', 'both']:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    if mode in ['test', 'both']:
        trainer = Trainer(accelerator="gpu", callbacks=[], logger=False, enable_checkpointing=False) # don't save checkpoints
        if ckpt_path is None:
            version_dirs = sorted(glob.glob(os.path.join(root_log_dir, "version_*")), key=os.path.getmtime)
            latest_version_dir = version_dirs[-1] if version_dirs else None
            checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
            best_checkpoint = glob.glob(f"{checkpoint_dir}/best-epoch=*.ckpt")
            if best_checkpoint:
                best_checkpoint_path = best_checkpoint[0]
                print(f"Best checkpoint path: {best_checkpoint_path}")
                best_model = DCBMInterface.load_from_checkpoint(best_checkpoint_path)
                trainer.test(best_model, data_module, ckpt_path=ckpt_path)
            else:
                print("No best checkpoint found in the latest version.")
        else:
            trainer.test(model, data_module, ckpt_path=ckpt_path)
    
if __name__ == '__main__':
    args = get_args()
    train(args)