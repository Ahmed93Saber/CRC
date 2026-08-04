import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, balanced_accuracy_score
from src.utils import update_confusion_matrix
import torch.nn.functional as F
from millab.src.models.clam import CLAMModel


def train_one_epoch(model, train_loader, criterion, optimizer, device, current_smoothing_matrix=None, epoch_cm=None):
    """
    Performs one epoch of training, with support for dynamic CPLS targets.
    """
    model.train()
    running_loss = 0.0

    for features, labels, _ in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Determine if we are using dynamic soft targets for CPLS
        if current_smoothing_matrix is not None:
            soft_targets = current_smoothing_matrix.to(device)[labels]
        else:
            soft_targets = None

        optimizer.zero_grad()
        if isinstance(model.base_model, CLAMModel):
            outputs = model(features, labels, torch.nn.CrossEntropyLoss())
        else:
            outputs = model(features)
        if isinstance(outputs, tuple):
            outputs = outputs[0]['logits']

        # Pass both targets to the criterion (soft_targets will be None if not using CPLS)
        loss = criterion(outputs, labels, soft_targets=soft_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        # # Track predictions for CPLS updates
        # if epoch_cm is not None:
        #     preds = torch.argmax(outputs, dim=1)
        #     update_confusion_matrix(epoch_cm, labels.cpu(), preds.cpu())

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
            # features, labels = {'features': features.to(device)}, labels.to(device)
            features, labels = features.to(device), labels.to(device)
            if isinstance(model.base_model, CLAMModel):
                outputs = model(features, labels, torch.nn.CrossEntropyLoss())
            else:
                outputs = model(features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]['logits']

            loss = criterion(outputs, labels)

            # running_loss += loss.item() * features['features'].size(0)
            running_loss += loss.item() * features.size(0)

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
    num_classes = np.array(all_probs).shape[1]
    avg_method = 'binary' if num_classes == 2 else 'weighted'

    # Calculate dynamically based on class count
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=avg_method, zero_division=0)
    prec = precision_score(all_labels, all_preds, average=avg_method, zero_division=0)
    rec = recall_score(all_labels, all_preds, average=avg_method, zero_division=0)

    # Calculate AUC
    if num_classes == 2:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    else:
        auc = roc_auc_score(
            all_labels,
            np.array(all_probs),
            multi_class='ovr',  # Use One-vs-Rest strategy
            average='macro'  # Handle class imbalance
        )

    # Return as a dictionary for scalability
    return {
        'loss': avg_loss,
        'acc': acc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'ids': all_ids,
        'probs':all_probs
    }