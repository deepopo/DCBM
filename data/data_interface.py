import os
import torch
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils import instantiate_from_config
from data.data_utils import ImbalancedDatasetSampler, EmptyDataset
from data.celeba import generate_data
from data.mine import MINEDataset

from utils import save_activations, get_save_names
from utils import LABEL_FILES, get_targets_only
from utils import MultiEpochsDataLoader

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

class MINEDataInterface(pl.LightningDataModule):
    def __init__(self, train_data=None, test_data=None, batch_size=100000):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.train_data is not None:
            self.train_dataset = MINEDataset(*self.train_data)
        if self.test_data is not None:
            self.test_dataset = MINEDataset(*self.test_data)

    def train_dataloader(self):
        if self.train_data is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return None

    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size)
        return None

class VLMDInterface(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.dataset = data_config.dataset
        self.clip_name = data_config.clip_name
        self.backbone = data_config.backbone
        self.feature_layer = data_config.feature_layer
        self.activation_batch_size = data_config.activation_batch_size
        self.proj_batch_size = data_config.proj_batch_size
        self.save_dir = data_config.save_dir
        self.clip_cutoff = data_config.clip_cutoff
        self.num_workers = data_config.num_workers
        self.device = torch.device('cuda:0')
        self.preprocess()

    def preprocess(self):
        #save activations and get save_paths
        concept_set = "dataset/concept_sets/{}_filtered.txt".format(self.dataset)

        #get concept set
        cls_file = LABEL_FILES[self.dataset]
        with open(cls_file, "r") as f:
            classes = f.read().split("\n")
        with open(concept_set) as f:
            concepts = f.read().split("\n")

        d_train = self.dataset + "_train"
        d_val = self.dataset + "_val"
        for d_probe in [d_train, d_val]:
            save_activations(clip_name = self.clip_name, target_name = self.backbone, 
                                    target_layers = [self.feature_layer], d_probe = d_probe,
                                    concept_set = concept_set, batch_size = self.activation_batch_size, 
                                    device = self.device, pool_mode = "avg", save_dir = self.save_dir)
            
        target_save_name, clip_save_name, text_save_name = get_save_names(self.clip_name, self.backbone, 
                                                self.feature_layer, d_train, concept_set, "avg", self.save_dir)
        val_target_save_name, val_clip_save_name, text_save_name =  get_save_names(self.clip_name, self.backbone,
                                                self.feature_layer, d_val, concept_set, "avg", self.save_dir)
        
        #load features
        with torch.no_grad():
            self.target_features = torch.load(target_save_name, map_location="cpu").float()
            self.val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        
            image_features = torch.load(clip_save_name, map_location="cpu").float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)

            val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
            val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

            text_features = torch.load(text_save_name, map_location="cpu").float()
            text_features /= torch.norm(text_features, dim=1, keepdim=True)
            
            clip_features = image_features @ text_features.T
            # val_clip_features = val_image_features @ text_features.T

            del text_features#, val_clip_features

        #filter concepts not activating highly
        highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

        # for i, concept in enumerate(concepts):
        #     if highest[i]<=self.clip_cutoff:
        #         print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))

        concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>self.clip_cutoff]

        #save memory by recalculating 
        del clip_features
        with torch.no_grad():
            self.text_features = torch.load(text_save_name, map_location="cpu").float()[highest>self.clip_cutoff]
            self.text_features /= torch.norm(self.text_features, dim=1, keepdim=True)
        
            self.clip_features = image_features @ self.text_features.T
            self.val_clip_features = val_image_features @ self.text_features.T
            del image_features
        
        # self.val_clip_features = self.val_clip_features[:, highest>self.clip_cutoff]

        self.train_targets = get_targets_only(d_train)
        self.val_targets = get_targets_only(d_val)
        self.train_targets = torch.LongTensor(self.train_targets)
        self.val_targets = torch.LongTensor(self.val_targets)
        self.explicit_dim = len(concepts)
        self.num_class = len(classes)

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.target_features, self.clip_features, self.train_targets)
        self.val_dataset = TensorDataset(self.val_target_features, self.val_clip_features, self.val_targets)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return MultiEpochsDataLoader(self.train_dataset, 
                          batch_size=self.proj_batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.proj_batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.proj_batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=True,
                          pin_memory=True,
                          )
