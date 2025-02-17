from torch.utils.data import Dataset

class MINEDataset(Dataset):
    def __init__(self, c_explicit, c_implicit, c_truth, y_explicit, y_implicit, y_truth, W_explicit, b_explicit, W_implicit, b_implicit):
        self.c_explicit = c_explicit
        self.c_implicit = c_implicit
        self.c_truth = c_truth
        self.y_explicit = y_explicit
        self.y_implicit = y_implicit
        self.y_truth = y_truth
        self.W_explicit = W_explicit
        self.b_explicit = b_explicit
        self.W_implicit = W_implicit
        self.b_implicit = b_implicit

    def __len__(self):
        return len(self.c_explicit)

    def __getitem__(self, idx):
        return {
            'c_explicit': self.c_explicit[idx],
            'c_implicit': self.c_implicit[idx],
            'c_truth': self.c_truth[idx],
            'y_explicit': self.y_explicit[idx],
            'y_implicit': self.y_implicit[idx],
            'y_truth': self.y_truth[idx],
            'W_explicit': self.W_explicit,
            'b_explicit': self.b_explicit,
            'W_implicit': self.W_implicit,
            'b_implicit': self.b_implicit
        }