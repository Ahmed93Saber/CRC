import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import shutil
import optuna

from millab.src.builder import create_model
from src.utils import (EarlyStopping, CPLS_CombinedCostLoss, initialize_uniform_smoothing,
                       compute_cpls_matrix, update_confusion_matrix)
from src.engine import train_one_epoch, evaluate_model
from src.cost_matrices import get_cost_matrix

from src.models import ClassificationModel
from sklearn.utils.class_weight import compute_class_weight


def train_and_validate_fold(fold_idx, train_loader, val_loader, params, device, model_dir, trial=None, epochs=50):
    """Handles the training, validation, and early stopping for a single fold."""

    model = create_model('abmil.base_mammoth.conch_v15', num_classes=4).to(device).to(device)
    # model = ClassificationModel(
    #     moe_args=params['moe_args'],
    #     output_dim=params['output_dim'],
    #     n_heads=params['n_heads'],
    #     hidden_dim=params['hidden_dim'],
    # ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # --- Calculate Class Weights for Cross-Entropy ---
    # Extract labels from the current fold's training loader
    train_labels = []
    # If your dataset directly exposes labels (e.g., train_loader.dataset.dataset.labels),
    # you can use that instead of iterating to save time.
    for _, labels, _ in train_loader:
        train_labels.extend(labels.cpu().numpy())

    # Compute balanced weights: n_samples / (n_classes * np.bincount(y))
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Use standard Cross Entropy with the calculated class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    early_stopper = EarlyStopping(patience=8, delta=0.001)
    best_epoch_metrics = {'f1': 0, 'auc': 0, 'acc': 0, 'preds': [], 'truths': []}

    for epoch in range(epochs):
        # Call train_one_epoch without any smoothing or cost matrices
        _ = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_results = evaluate_model(model, val_loader, criterion, device)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs} | Val Loss: {val_results['loss']:.4f} | F1: {val_results['f1']:.4f} | AUC: {val_results['auc']:.4f}")

        # Track best metric
        if val_results['acc'] > (early_stopper.best_score if early_stopper.best_score else -np.inf):
            best_epoch_metrics.update({
                'f1': val_results['f1'],
                'auc': val_results['auc'],
                'preds': val_results['preds'],
                'truths': val_results['labels'],
                'acc': val_results['acc'],
            })

            ckpt_name = f"fold_{fold_idx}.pt"
            torch.save(model.state_dict(), os.path.join(model_dir, ckpt_name))

        # Optuna Pruning
        if trial is not None:
            trial.report(val_results['acc'], epoch + (fold_idx * epochs))
            if trial.should_prune():
                print(f"  [INFO] Trial pruned by Optuna at fold {fold_idx + 1}, epoch {epoch + 1}")
                raise optuna.TrialPruned()

        early_stopper(val_results['acc'])
        if early_stopper.early_stop:
            print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    return best_epoch_metrics


def evaluate_test_set(test_dataset, params,  device, model_dir, n_splits):
    """Loads saved fold models and runs inference across the test dataset."""
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    test_data = {
        'aucs': [], 'preds': [], 'truths': [], 'cms': [],
        'f1s': [], 'precs': [], 'recs': [], 'ids': [], 'accs': [],
    }

    for k in range(n_splits):
        model_path = os.path.join(model_dir, f"fold_{k}.pt")
        model = create_model('abmil.base_mammoth.conch_v15', num_classes=4).to(device).to(device)
        # model = ClassificationModel(
        #     moe_args=params['moe_args'],
        #     output_dim=params['output_dim'],
        #     n_heads=params['n_heads'],
        #     hidden_dim=params['hidden_dim'],
        #     encoder_type=params['encoder_type'],
        # ).to(device)
        model.load_state_dict(torch.load(model_path))

        # Catch the dictionary
        test_results = evaluate_model(model, test_loader, criterion, device)

        test_data['f1s'].append(test_results['f1'])
        test_data['aucs'].append(test_results['auc'])
        test_data['accs'].append(test_results['acc'])
        test_data['precs'].append(test_results['prec'])
        test_data['recs'].append(test_results['rec'])
        test_data['preds'].append(test_results['preds'])
        test_data['truths'].append(test_results['labels'])
        test_data['ids'].append(test_results['ids'])
        test_data['cms'].append(confusion_matrix(test_results['labels'], test_results['preds']))

    return test_data


def save_optuna_artifacts(save_dir, test_data):
    """Helper function to save numpy arrays of our test predictions."""

    def save_obj(name, data):
        np.save(f"{save_dir}/{name}.npy", np.array(data, dtype=object))

    np.save(f"{save_dir}/confusion_matrices.npy", np.array(test_data['cms']))
    save_obj("predictions", test_data['preds'])
    save_obj("ground_truths", test_data['truths'])
    save_obj("file_ids", test_data['ids'])


def save_validation_artifacts(save_dir, val_data, val_ids):
    """Helper function to save validation predictions, labels, and file IDs for each fold."""
    def save_obj(name, data):
        np.save(f"{save_dir}/{name}.npy", np.array(data, dtype=object))

    save_obj("val_predictions", val_data['preds'])
    save_obj("val_ground_truths", val_data['truths'])
    save_obj("val_file_ids", val_ids)


def run_cross_validation(datasets, params, device, trial=None, n_splits=5, epochs=50):
    """Main Orchestrator: Runs stratified K-Fold cross validation and test set evaluation."""
    print(f"Starting {n_splits}-Fold Stratified CV ({n_splits} total runs)...")

    save_dir = f"./artifacts/{params['exp_name']}/trial_{trial.number}" if trial else f"./artifacts_max_{params['exp_name']}/default"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    label_col_name = params['label_col']
    labels_list = datasets['train'].df[label_col_name].values
    dummy_X = np.zeros(len(labels_list))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_overall_f1s = []
    fold_overall_aucs = []
    fold_overall_accs = []
    val_data = {
        'preds': [], 'truths': []
    }

    for current_fold, (train_ids, val_ids) in enumerate(skf.split(dummy_X, labels_list)):
        print(f"\n--- Run {current_fold + 1}/{n_splits} | Fold {current_fold + 1} ---")

        train_loader = DataLoader(
            datasets['train'],
            batch_size=params['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(train_ids)
        )
        val_dataset_fold = Subset(datasets['val'], val_ids)
        val_loader = DataLoader(val_dataset_fold, batch_size=1, shuffle=False)

        best_epoch_metrics = train_and_validate_fold(
            fold_idx=current_fold,
            train_loader=train_loader,
            val_loader=val_loader,
            params=params,
            device=device,
            model_dir=model_dir,
            trial=trial,
            epochs=epochs
        )

        fold_overall_f1s.append(best_epoch_metrics['f1'])
        fold_overall_aucs.append(best_epoch_metrics['auc'])
        fold_overall_accs.append(best_epoch_metrics['acc'])

        # Collect validation predictions and labels for each fold
        val_data['preds'].append(best_epoch_metrics['preds'])
        val_data['truths'].append(best_epoch_metrics['truths'])

    avg_f1_overall = np.mean(fold_overall_f1s)
    avg_auc_overall = np.mean(fold_overall_aucs)
    avg_acc_overall = np.mean(fold_overall_accs)

    if trial is not None:
        trial.set_user_attr("avg_F1_val", float(avg_f1_overall))
        trial.set_user_attr("avg_auc_val", float(avg_auc_overall))
        trial.set_user_attr("avg_acc_val", float(avg_acc_overall))

        if avg_auc_overall > 0.85 and avg_f1_overall > 0.85:
            print(
                f"  [INFO] Validation targets met (AUC: {avg_auc_overall:.4f}, F1: {avg_f1_overall:.4f}). Evaluating Test Set...")

            # Save validation predictions and labels
            save_validation_artifacts(save_dir, val_data, val_ids)

            test_data = evaluate_test_set(datasets['test'], params, device, model_dir, n_splits)

            avg_test_f1 = np.mean(test_data['f1s'])
            avg_test_auc = np.mean(test_data['aucs'])
            avg_test_acc = np.mean(test_data['accs'])
            print(f"Avg acc: {avg_test_acc:.4f}, Avg auc: {avg_test_auc:.4f}, Avg f1: {avg_test_f1:.4f}")

            trial.set_user_attr("avg_F1_test", float(avg_test_f1))
            trial.set_user_attr("avg_auc_test", float(avg_test_auc))
            trial.set_user_attr("avg_acc_test", float(avg_test_acc))
            trial.set_user_attr("avg_precision_test", float(np.mean(test_data['precs'])))
            trial.set_user_attr("avg_recall_test", float(np.mean(test_data['recs'])))

            if avg_test_auc > 0.9 and avg_test_f1 > 0.8:
                print(
                    f"  [SUCCESS] Test targets met (AUC: {avg_test_auc:.4f}, F1: {avg_test_f1:.4f}). Models retained in {model_dir}")
                save_optuna_artifacts(save_dir, test_data)
            else:
                print(
                    f"  [INFO] Test targets NOT met (AUC: {avg_test_auc:.4f}, F1: {avg_test_f1:.4f}). Deleting models...")
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
        else:
            print(f"  [INFO] Validation targets NOT met. Skipping test evaluation and deleting models...")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

    print(f"\nOverall Validation Results (across {n_splits} folds):")
    print(f"  Avg Val F1:  {avg_f1_overall:.4f}")
    print(f"  Avg Val AUC: {avg_auc_overall:.4f}")
    print(f"  Avg Val ACC: {avg_acc_overall:.4f}")


    return avg_acc_overall