import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from src.utils import calculate_metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    # Unpack 4 items: features, labels, ids, pdl1
    # We ignore ids and pdl1 during training using '_'
    for features, labels, _, _ in train_loader:
        features, labels = {'features': features.to(device)}, labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features['features'].size(0)

    return running_loss / len(train_loader.dataset)


def evaluate_model(model, val_loader, criterion, device):
    """
    Evaluates the model and returns a dictionary of all metrics and metadata.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for features, labels, file_ids in val_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features['features'].size(0)

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(file_ids)

    avg_loss = running_loss / len(val_loader.dataset)

    # Calculate basic metrics
    bal_acc, f1 = calculate_metrics(np.array(all_labels), np.array(all_preds))
    num_classes = np.array(all_probs).shape[1]

    # Calculate Precision and Recall
    avg_method = 'binary' if num_classes == 2 else 'weighted'
    prec = precision_score(all_labels, all_preds, average=avg_method, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=avg_method, zero_division=0)

    # Calculate AUC
    if num_classes == 2:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    else:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')

    # Return as a dictionary for scalability
    return {
        'loss': avg_loss,
        'bal_acc': bal_acc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'ids': all_ids
    }