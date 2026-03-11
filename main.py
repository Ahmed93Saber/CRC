import optuna
import os
from src.datasets import H5Dataset
from src.train import run_cross_validation
from src.utils import get_device, seed_everything

# --- Configuration ---
CSV_PATH_TRAIN = './dataframes/structured_labels_biopsy_final_NEW.csv'
CSV_PATH_TEST = './dataframes/annotations_all_HunCRC_NEW.csv'
H5_DIR_TRAIN = './features_conch_v15_CAL'
H5_DIR_TEST = './features_conch_v15_HUN'
LABEL_COL = 'label'
ID_COL = 'slide'
INPUT_DIM = 768
OUTPUT_DIM = 4  # e.g., Multi-class
N_FOLDS = 5
MAX_EPOCHS = 200


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
        csv_path=CSV_PATH_TRAIN,
        feats_path=H5_DIR_TRAIN,
        label_col=LABEL_COL,
        split='train',
        id_col=ID_COL
    )

    val_dataset = H5Dataset(
        csv_path=CSV_PATH_TRAIN,
        feats_path=H5_DIR_TRAIN,
        label_col=LABEL_COL,
        split='val',
        id_col=ID_COL
    )

    test_dataset = H5Dataset(
        csv_path=CSV_PATH_TEST,
        feats_path=H5_DIR_TEST,
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
