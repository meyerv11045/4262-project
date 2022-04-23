import torch
import numpy as np

class DateDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        super().__init__()
        self.data = np.genfromtxt(file, dtype=np.float32, delimiter=",", skip_header=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :-2], np.array(self.data[idx, -1])