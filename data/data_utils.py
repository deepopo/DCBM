import torch
import pickle as pkl
from torch.utils.data import Dataset

from utils.config import concept_samples_10p, concept_samples_20p, concept_samples_30p, concept_samples_40p, concept_samples_50p, derm_concept_samples_25p, derm_concept_samples_50p

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples

def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1, concept_percent=100):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    concept_percent: the ratio of concepts that are left
    """
    imbalance_ratio = []
    data = pkl.load(open(pkl_file, 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label']) * concept_percent // 100
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if concept_percent == 50:
            if n_attr > 50:
                # CUB
                labels = [labels[index] for index in concept_samples_50p]
            else:
                # Derm7pt
                labels = [labels[index] for index in derm_concept_samples_50p]
        elif concept_percent == 40:
            labels = [labels[index] for index in concept_samples_40p]
        elif concept_percent == 30:
            labels = [labels[index] for index in concept_samples_30p]
        elif concept_percent == 20:
            labels = [labels[index] for index in concept_samples_20p]
        elif concept_percent == 10:
            labels = [labels[index] for index in concept_samples_10p]
        elif concept_percent == 25:
            labels = [labels[index] for index in derm_concept_samples_25p]
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio