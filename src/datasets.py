import h5py
import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class H5Dataset(Dataset):
    def __init__(self, feats_path: str, csv_path: str, id_col: str,
                 label_col: str, split: str = 'train', num_features=512,
                 is_aug=False, dropout_p=0.1, aug_prob=0.25):

        self.df = pd.read_csv(csv_path)
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        self.id_col = id_col
        self.label_col = label_col
        # Store the augmentation flag
        self.is_aug = is_aug
        self.dropout_p = dropout_p
        self.aug_prob = aug_prob

        # 1. Strip spaces from the entire column at once
        # self.df[self.id_col] = self.df[self.id_col].astype(str).str.replace(" ", "")

        # 2. Apply the zfill logic vectorized based on the split
        if self.split in ('train', 'val'):
            # Find the rows where cohort is CAL and pad them
            cal_mask = self.df['cohort'] == 'CAL'
            self.df.loc[cal_mask, self.id_col] = self.df.loc[cal_mask, self.id_col]   # .str.replace(" ", "")
            self.df.loc[cal_mask, self.id_col] = self.df.loc[cal_mask, self.id_col].str.zfill(18)

        elif self.split == 'test':
            # Pad the whole column for the test split
            self.df[self.id_col] = self.df[self.id_col].astype(str).str.zfill(3)


    def __len__(self):
        return len(self.df)

    def _apply_augmentations(self, features, noise_std=0.01, dropout_p=0.05, aug_prob=0.25):
        """Applies embedding-level augmentations to the feature tensor with a given probability."""

        # Randomly decide whether to skip augmentations for this specific batch/bag
        if torch.rand(1).item() > aug_prob:
            return features

        # # 1. Gaussian Noise
        # noise = torch.randn_like(features) * noise_std
        # features = features + noise

        # 2. Feature Dropout
        dropout_mask = (torch.rand_like(features) > dropout_p).float()
        features = (features * dropout_mask) / (1.0 - dropout_p)

        return features

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row[self.id_col])



        with h5py.File(os.path.join(self.feats_path, file_id + '.h5'), "r") as f:
            features = torch.from_numpy(f["features"][:])

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator())[
                    :self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,),
                                        generator=torch.Generator())  # Oversampling
            features = features[indices]

            # --- AUGMENTATIONS ---
            if self.is_aug:
                features = self._apply_augmentations(features, dropout_p=self.dropout_p, aug_prob=self.aug_prob)

        label_map = {'low-grade dysplasia': 0, 'high-grade dysplasia': 1, 'adenocarcinoma': 2, 'other': 3}

        # Use .get() with a default that will throw a clear error if unmapped,
        # but ensure the default isn't passed directly into torch.tensor if it's a string
        label_idx = label_map.get(row[self.label_col])
        if label_idx is None:
            raise ValueError(f"Unidentified Label: {row[self.label_col]}")

        label = torch.tensor(label_idx, dtype=torch.long)

        return features, label, file_id