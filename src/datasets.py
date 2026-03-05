import h5py
import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class H5Dataset(Dataset):
    def __init__(self, feats_path: str, csv_path: str, id_col: str,
                 label_col: str,  split: str = 'train', num_features=512):
        self.df = pd.read_csv(csv_path)
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        self.id_col = id_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row[self.id_col]).replace(" ", "")
        pdl1_status = str(row["PD-L1 (TPS Score)"]).replace(" ", "")

        with h5py.File(os.path.join(self.feats_path, file_id + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator())[
                    :self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator()) # Oversampling
            features = features[indices]

        label = torch.tensor(row[self.label_col], dtype=torch.long)
        return features, label, file_id
