import torch
import numpy as np
import random
import os
from sklearn.metrics import balanced_accuracy_score, f1_score


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred):
    """
    Calculates Balanced Accuracy and Weighted F1 Score.
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
    Returns:
        tuple: (balanced_accuracy, f1_score)
    """
    # average='weighted' accounts for class imbalance
    # average='macro' treats all classes equally regardless of size
    f1 = f1_score(y_true, y_pred, average='weighted')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return bal_acc, f1

class EarlyStopping:
    def __init__(self, patience=20, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """
        Check if we should stop.
        Note: We are assuming we want to MAXIMIZE the score (F1).
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
