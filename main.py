import optuna
import os
from src.datasets import H5Dataset
from src.train import run_cross_validation
from src.utils import get_device, seed_everything

# --- Configuration ---
CSV_PATH = './dataframes/metadata.csv'
H5_DIR = './features_conch_v15_no_holes'
LABEL_COL = 'label-1-si'
ID_COL = 'SLIDE_ID'
INPUT_DIM = 768
OUTPUT_DIM = 2  # e.g., Multi-class (2 classes)
N_FOLDS = 5
MAX_EPOCHS = 100  # Increased since we have early stopping


def objective(trial):
    device = get_device()
    seed_everything(42)

    # Define Hyperparameters
    params = {
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'label_col': LABEL_COL,
        'n_layers': trial.suggest_int('n_layers', 3, 5),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]),
        'n_heads': trial.suggest_categorical('n_heads', [1, 2, 3, 4, 6, 8]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16])
    }

    train_dataset = H5Dataset(
        csv_path=CSV_PATH,
        feats_path=H5_DIR,
        label_col=LABEL_COL,
        split='train',
        id_col=ID_COL
    )

    val_dataset = H5Dataset(
        csv_path=CSV_PATH,
        feats_path=H5_DIR,
        label_col=LABEL_COL,
        split='val',
        id_col=ID_COL
    )

    test_dataset = H5Dataset(
        csv_path=CSV_PATH,
        feats_path=H5_DIR,
        label_col=LABEL_COL,
        split='test',
        id_col=ID_COL
    )

    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    # Run CV and get Average F1
    avg_f1 = run_cross_validation(
        datasets=datasets,
        params=params,
        device=device,
        trial=trial,
        n_splits=N_FOLDS,
        epochs=MAX_EPOCHS
    )

    return avg_f1


if __name__ == "__main__":
    # Direction is MAXIMIZE because we are returning F1 Score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    optuna_df_path = "optuna_trials.csv"
    study.trials_dataframe().to_csv(optuna_df_path, index=False)

    print("\n--- Optimization Finished ---")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Avg F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
