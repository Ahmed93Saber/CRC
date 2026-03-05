import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import shutil

from src.models import BinaryClassificationModel
from src.utils import EarlyStopping
from src.engine import train_one_epoch, evaluate_model


def run_cross_validation(datasets, params, device, trial=None, n_splits=5, epochs=50):
    total_runs = n_splits

    print(f"Starting {n_splits}-Fold Stratified CV ({total_runs} total runs)...")

    # --- Setup Final Directories directly ---
    save_dir = f"./artifacts_max/trial_{trial.number}" if trial else "./artifacts_max/default"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)  # Create the model directory immediately

    # --- Setup Data Splits ---
    label_col_name = params['label_col']
    labels_list = datasets['train'].df[label_col_name].values
    dummy_X = np.zeros(len(labels_list))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # --- Tracking Structures ---
    fold_overall_f1s = []
    fold_overall_aucs = []

    cv_data = {
        'aucs': [], 'preds': [], 'truths': [], 'cms': [],
        'f1s': [], 'precs': [], 'recs': [], 'ids': [], 'pdl1s': []
    }

    # --- Main Loop ---
    for current_fold, (train_ids, val_ids) in enumerate(skf.split(dummy_X, labels_list)):

        print(f"\n--- Run {current_fold + 1}/{total_runs} | Fold {current_fold + 1} ---")

        # 1. Setup Model & Optimization
        model = BinaryClassificationModel(
            output_dim=params['output_dim'],
            n_heads=params['n_heads'],
            hidden_dim=params['hidden_dim'],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        # 2. Setup Loaders
        train_loader = DataLoader(datasets['train'], batch_size=params['batch_size'],
                                  sampler=SubsetRandomSampler(train_ids))
        val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False, sampler=SubsetRandomSampler(val_ids))

        # 3. Training Phase
        early_stopper = EarlyStopping(patience=10, delta=0.001)

        best_fold = {
            'f1': 0, 'auc': 0, 'prec': 0, 'rec': 0,
            'preds': [], 'truths': [],
            'ids': [], 'pdl1s': []
        }

        for epoch in range(epochs):
            _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_bal_acc, val_f1, val_prec, val_rec, val_auc, preds, truths, ids, pdl1s = evaluate_model(model,
                                                                                                                  val_loader,
                                                                                                                  criterion,
                                                                                                                  device)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

            # Track best F1
            if val_f1 > (early_stopper.best_score if early_stopper.best_score else -np.inf):
                best_fold.update({
                    'f1': val_f1, 'auc': val_auc, 'prec': val_prec, 'rec': val_rec,
                    'preds': preds, 'truths': truths,
                    'ids': ids, 'pdl1s': pdl1s
                })

                # Save directly to the final models directory
                ckpt_name = f"fold_{current_fold}.pt"
                torch.save(model.state_dict(), os.path.join(model_dir, ckpt_name))

            early_stopper(val_f1)
            if early_stopper.early_stop:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

        # 4. Store Fold Results
        fold_overall_f1s.append(best_fold['f1'])
        fold_overall_aucs.append(best_fold['auc'])

        # Append to overall CV Data
        cv_data['f1s'].append(best_fold['f1'])
        cv_data['aucs'].append(best_fold['auc'])
        cv_data['precs'].append(best_fold['prec'])
        cv_data['recs'].append(best_fold['rec'])
        cv_data['preds'].append(best_fold['preds'])
        cv_data['truths'].append(best_fold['truths'])
        cv_data['ids'].append(best_fold['ids'])
        cv_data['pdl1s'].append(best_fold['pdl1s'])
        cv_data['cms'].append(confusion_matrix(best_fold['truths'], best_fold['preds']))

    # --- Aggregation & Saving ---
    avg_f1_overall = np.mean(fold_overall_f1s)
    avg_auc_overall = np.mean(fold_overall_aucs)
    avg_prec_overall = np.mean(cv_data['precs'])
    avg_rec_overall = np.mean(cv_data['recs'])

    # Optuna Logging & Artifact Saving
    if trial is not None:
        def save_obj(name, data):
            np.save(f"{save_dir}/{name}.npy", np.array(data, dtype=object))

        # Save Metrics
        np.save(f"{save_dir}/confusion_matrices.npy", np.array(cv_data['cms']))
        save_obj("predictions", cv_data['preds'])
        save_obj("ground_truths", cv_data['truths'])
        save_obj("file_ids", cv_data['ids'])
        save_obj("pdl1_status", cv_data['pdl1s'])

        trial.set_user_attr("artifact_dir", save_dir)
        trial.set_user_attr("avg_F1_overall", float(avg_f1_overall))
        trial.set_user_attr("avg_auc_overall", float(avg_auc_overall))
        trial.set_user_attr("avg_precision_overall", float(avg_prec_overall))
        trial.set_user_attr("avg_recall_overall", float(avg_rec_overall))

        # Check Threshold to determine if we KEEP the saved models
        if avg_auc_overall > 0.69 and avg_f1_overall > 0.75:
            print(f"  [INFO] Overall AUC ({avg_auc_overall:.4f}) > 0.69 and F1 > 0.75. Models retained in {model_dir}")
        else:
            print(f"  [INFO] Targets not met. Deleting sub-par models...")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)  # Delete the bad models directly

    print(f"\nOverall Results (across {total_runs} folds):")
    print(f"  Avg F1:  {avg_f1_overall:.4f}")
    print(f"  Avg AUC: {avg_auc_overall:.4f}")


    return avg_f1_overall