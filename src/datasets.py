import h5py
import os
import torch
import random
from torch.utils.data import Dataset
import pandas as pd


class H5Dataset(Dataset):
    def __init__(self, feats_path: str, csv_path: str, id_col: str,
                 label_col: str, split: str = 'train', num_features=4096,
                 aug_feats_path: str = None, aug_prob: float = 0.25):

        self.df = pd.read_csv(csv_path)
        self.feats_path = feats_path
        self.aug_feats_path = aug_feats_path
        self.aug_prob = aug_prob
        self.num_features = num_features
        self.split = split
        self.id_col = id_col
        self.label_col = label_col


        # 1. Strip spaces from the entire column at once
        # self.df[self.id_col] = self.df[self.id_col].astype(str).str.replace(" ", "")

        # 2. Apply the zfill logic vectorized based on the split
        if self.split in ('train', 'val'):
            # Find the rows where cohort is CAL and pad them
            cal_mask = self.df['cohort'] == 'CAL'
            self.df.loc[cal_mask, self.id_col] = self.df.loc[cal_mask, self.id_col].astype(str)   # .str.replace(" ", "")
            self.df.loc[cal_mask, self.id_col] = self.df.loc[cal_mask, self.id_col].str.zfill(18)

        elif self.split == 'test':
            # Pad the whole column for the test split
            self.df[self.id_col] = self.df[self.id_col].astype(str).str.zfill(3)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row[self.id_col])

        # --- AUGMENTATION LOGIC ---
        # Default to the original un-augmented features directory
        current_feats_path = self.feats_path

        # Only swap paths if we are in training mode AND an aug path was provided
        if self.split == 'train' and self.aug_feats_path is not None:
            # Flip a coin
            if random.random() < self.aug_prob:
                # Safety check: Make sure the augmented file actually exists before using the path
                if os.path.exists(os.path.join(self.aug_feats_path, file_id + '.h5')):
                    current_feats_path = self.aug_feats_path
        try:
            with h5py.File(os.path.join(current_feats_path, file_id + '.h5'), "r") as f:
                features = torch.from_numpy(f["features"][:])
        except Exception:
            print("STOP")

        if self.split == 'train':
            try:
                num_available = features.shape[0]
            except Exception as e:
                print(e)
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator())[
                    :self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,),
                                        generator=torch.Generator())  # Oversampling
            features = features[indices]


        label_map = {'low-grade dysplasia': 0, 'high-grade dysplasia': 1, 'adenocarcinoma': 2, 'other': 3}

        # Use .get() with a default that will throw a clear error if unmapped,
        # but ensure the default isn't passed directly into torch.tensor if it's a string
        label_idx = label_map.get(row[self.label_col])
        if label_idx is None:
            raise ValueError(f"Unidentified Label: {row[self.label_col]}")

        label = torch.tensor(label_idx, dtype=torch.long)

        return features, label, file_id