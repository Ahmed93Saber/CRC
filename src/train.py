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

from src.models import BinaryClassificationModel
from src.utils import EarlyStopping
from src.engine import train_one_epoch, evaluate_model


def train_and_validate_fold(fold_idx, train_loader, val_loader, params, device, model_dir, trial=None, epochs=50):
    """Handles the training, validation, and early stopping for a single fold."""
    # 1. Setup Model & Optimization
    model = BinaryClassificationModel(
        output_dim=params['output_dim'],
        n_heads=params['n_heads'],
        hidden_dim=params['hidden_dim'],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=10, delta=0.001)

    best_epoch_results = {'f1': 0, 'auc': 0, 'preds': [], 'truths': []}

    for epoch in range(epochs):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_bal_acc, val_f1, _, _, val_auc, preds, truths, _ = evaluate_model(
            model, val_loader, criterion, device
        )

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        # Track best F1
        if val_f1 > (early_stopper.best_score if early_stopper.best_score else -np.inf):
            best_epoch_results.update({
                'f1': val_f1, 'auc': val_auc,
                'preds': preds, 'truths': truths
            })

            # Save directly to the final models directory
            ckpt_name = f"fold_{fold_idx}.pt"
            torch.save(model.state_dict(), os.path.join(model_dir, ckpt_name))

        # Optuna Pruning
        if trial is not None:
            trial.report(val_f1, epoch + (fold_idx * epochs))
            if trial.should_prune():
                print(f"  [INFO] Trial pruned by Optuna at fold {fold_idx + 1}, epoch {epoch + 1}")
                raise optuna.TrialPruned()

        early_stopper(val_f1)
        if early_stopper.early_stop:
            print(f"  Early stopping triggered at epoch {epoch + 1}")
            break

    return best_epoch_results


def evaluate_test_set(test_dataset, params, device, model_dir, n_splits):
    """Loads saved fold models and runs inference across the test dataset."""
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    test_data = {
        'aucs': [], 'preds': [], 'truths': [], 'cms': [],
        'f1s': [], 'precs': [], 'recs': [], 'ids': []
    }

    for k in range(n_splits):
        model_path = os.path.join(model_dir, f"fold_{k}.pt")
        model = BinaryClassificationModel(
            output_dim=params['output_dim'],
            n_heads=params['n_heads'],
            hidden_dim=params['hidden_dim'],
        ).to(device)
        model.load_state_dict(torch.load(model_path))

        _, _, test_f1, test_prec, test_rec, test_auc, t_preds, t_truths, t_ids = evaluate_model(
            model, test_loader, criterion, device
        )

        test_data['f1s'].append(test_f1)
        test_data['aucs'].append(test_auc)
        test_data['precs'].append(test_prec)
        test_data['recs'].append(test_rec)
        test_data['preds'].append(t_preds)
        test_data['truths'].append(t_truths)
        test_data['ids'].append(t_ids)
        test_data['cms'].append(confusion_matrix(t_truths, t_preds))

    return test_data


def save_optuna_artifacts(save_dir, test_data):
    """Helper function to save numpy arrays of our test predictions."""

    def save_obj(name, data):
        np.save(f"{save_dir}/{name}.npy", np.array(data, dtype=object))

    np.save(f"{save_dir}/confusion_matrices.npy", np.array(test_data['cms']))
    save_obj("predictions", test_data['preds'])
    save_obj("ground_truths", test_data['truths'])
    save_obj("file_ids", test_data['ids'])


def run_cross_validation(datasets, params, device, trial=None, n_splits=5, epochs=50):
    """Main Orchestrator: Runs stratified K-Fold cross validation and test set evaluation."""
    print(f"Starting {n_splits}-Fold Stratified CV...")

    # --- Setup Final Directories directly ---
    save_dir = f"./artifacts_max/trial_{trial.number}" if trial else "./artifacts_max/default"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # --- Setup Data Splits ---
    label_col_name = params['label_col']
    labels_list = datasets['train'].df[label_col_name].values
    dummy_X = np.zeros(len(labels_list))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Tracking Structures (Validation) ---
    fold_overall_f1s = []
    fold_overall_aucs = []

    # --- Main Validation Loop ---
    for current_fold, (train_ids, val_ids) in enumerate(skf.split(dummy_X, labels_list)):
        print(f"\n--- Run {current_fold + 1}/{n_splits} | Fold {current_fold + 1} ---")

        train_loader = DataLoader(
            datasets['train'],
            batch_size=params['batch_size'],
            sampler=torch.utils.data.SubsetRandomSampler(train_ids)
        )
        val_dataset_fold = Subset(datasets['val'], val_ids)  # use PyTorch's Subset for val_loader so it's deterministic
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

    # --- Aggregation & Test Phase ---
    avg_f1_overall = np.mean(fold_overall_f1s)
    avg_auc_overall = np.mean(fold_overall_aucs)

    if trial is not None:
        # Log validation metrics
        trial.set_user_attr("avg_F1_val", float(avg_f1_overall))
        trial.set_user_attr("avg_auc_val", float(avg_auc_overall))

        # Condition 1: Check Validation Threshold
        if avg_auc_overall > 0.69 and avg_f1_overall > 0.75:
            print(
                f"  [INFO] Validation targets met (AUC: {avg_auc_overall:.4f}, F1: {avg_f1_overall:.4f}). Evaluating Test Set...")

            test_data = evaluate_test_set(datasets['test'], params, device, model_dir, n_splits)

            avg_test_f1 = np.mean(test_data['f1s'])
            avg_test_auc = np.mean(test_data['aucs'])

            trial.set_user_attr("avg_F1_test", float(avg_test_f1))
            trial.set_user_attr("avg_auc_test", float(avg_test_auc))
            trial.set_user_attr("avg_precision_test", float(np.mean(test_data['precs'])))
            trial.set_user_attr("avg_recall_test", float(np.mean(test_data['recs'])))

            # Condition 2: Check Test Threshold
            if avg_test_auc > 0.7 and avg_test_f1 > 0.7:
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

    return avg_f1_overall