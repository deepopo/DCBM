"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from utils.config import derm_concept_samples_25p, derm_concept_samples_50p
from torch.utils.data import Dataset

class DermDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the Derm7pt dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label,
                 n_class_attr, resol, n_attributes, concept_percent):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        resol: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        concept_percent: the ratio of concepts that are left
        """
        self.n_attributes = n_attributes
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        if self.is_train:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.n_class_attr = n_class_attr
        self.concept_percent = concept_percent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        index = img_path.split('/').index('release_v0')
        img_path = os.path.join('dataset/', '/'.join(img_path.split('/')[index:]))
        img = Image.open(img_path).convert('RGB')
        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
                if self.concept_percent == 25:
                    attr_label = [attr_label[index] for index in derm_concept_samples_25p]
                elif self.concept_percent == 50:
                    attr_label = [attr_label[index] for index in derm_concept_samples_50p]
                elif self.concept_percent != 100:
                    print("Current percent is not supported. We use all concepts instead.")
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((self.n_attributes, self.n_class_attr))
                    one_hot_attr_label[np.arange(self.n_attributes), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label


