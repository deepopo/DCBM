import os
import torch
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

from utils import instantiate_from_config
from data.data_utils import ImbalancedDatasetSampler, EmptyDataset
from data.celeba import generate_data

class DInterface(pl.LightningDataModule):
    def __init__(self, data_config, data_dir, batch_size):
        super().__init__()
        self.num_workers = data_config.num_workers
        self.batch_size = batch_size

        data_config.params.pkl_file_paths = [os.path.join(data_dir, path) for path in data_config.train_pkl_paths]
        self.train_dataset = instantiate_from_config(data_config)
        self.train_sampler = BatchSampler(ImbalancedDatasetSampler(self.train_dataset), batch_size=batch_size, drop_last=True)
        if len(data_config.val_pkl_paths) != 0:
            data_config.params.pkl_file_paths = [os.path.join(data_dir, path) for path in data_config.val_pkl_paths]
            self.val_dataset = instantiate_from_config(data_config)
        else:
            self.val_dataset = None
        data_config.params.pkl_file_paths = [os.path.join(data_dir, path) for path in data_config.test_pkl_paths]
        self.test_dataset = instantiate_from_config(data_config)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.trainset = self.train_dataloader()
        self.valset = self.val_dataloader()

        # Assign test dataset for use in dataloader(s)
        self.testset = self.test_dataloader()

    def train_dataloader(self):
        # return DataLoader(self.train_dataset,
        #                   batch_sampler=self.train_sampler, 
        #                   num_workers=self.num_workers)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_dataset is None:
            return DataLoader(EmptyDataset(), batch_size=self.batch_size)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False, 
                          num_workers=self.num_workers)

class CelebAInterface(pl.LightningDataModule):
    def __init__(self, data_config, data_dir, save_dir, batch_size):
        super().__init__()
        self.num_workers = data_config.num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir        
        self.save_dir = save_dir
        self.resol = data_config.params.resol
        self.preprocess()

    def preprocess(self):
        self.trainset, self.valset, self.testset, imbalance = generate_data(root_dir=self.data_dir, batch_size=self.batch_size, resol=self.resol, num_workers=self.num_workers)
        imbalance_path = os.path.join(self.save_dir, 'celeba_imbalance.pth')
        if not os.path.exists(imbalance_path):
            with open(imbalance_path, 'wb') as f:
                pkl.dump(imbalance, f)
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.trainset = self.train_dataloader()
        self.valset = self.val_dataloader()

        # Assign test dataset for use in dataloader(s)
        self.testset = self.test_dataloader()

    def train_dataloader(self):
        return self.trainset

    def val_dataloader(self):
        return self.valset

    def test_dataloader(self):
        return self.testset