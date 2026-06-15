import optuna
import os
from src.datasets import H5Dataset
from src.train import run_cross_validation
from src.utils import get_device, seed_everything
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*non-tuple sequence for multidimensional indexing.*")

# --- Configuration ---
CSV_PATH_TRAIN = './dataframes/combined_cohorts_CAL+BAY.csv'
CSV_PATH_TEST = './dataframes/annotations_all_HunCRC_NEW.csv'
H5_DIR_TRAIN = 'features/features_conch_v15_CAL'
# H5_DIR_TRAIN = r"W:\pathologie\bioinfo-archive\TridentPipelineOutput\CRC\CaltagironeUNIV2\20x_256px_0px_overlap\features_uni_v2"
H5_DIR_TEST = 'features/features_conch_v15_HUN'
# H5_DIR_TEST = r"W:\pathologie\bioinfo-archive\TridentPipelineOutput\CRC\HunCRCUNIV2\20x_256px_0px_overlap\features_uni_v2"
LABEL_COL = 'label'
ID_COL = 'slide'
INPUT_DIM = 768
OUTPUT_DIM = 4  # e.g., Multi-class
N_FOLDS = 5
MAX_EPOCHS = 200
AUG_h5_DIR = "features/features_conch_v15_CAL_AUG"








def objective(trial):
    device = get_device()
    seed_everything(42)

    # Define Hyperparameters
    params = {
        'exp_name': EXP_NAME,
        'input_dim': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
        'label_col': LABEL_COL,
        'n_layers': trial.suggest_int('n_layers', 3, 5),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [512, 1024]),
        'n_heads': trial.suggest_categorical('n_heads', [3, 4, 6]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16]),
        'loss_beta': 0, # trial.suggest_float('loss_beta', 0.05, 1.0, log=True),
        'cpls_alpha': 0, # No CPLS trial.suggest_float('cpls_alpha', 0.01, 0.15, log=True),
        'matrix_name': trial.suggest_categorical('matrix_name', ["asymmetric_risk",
                                                                "squared_distance"]),
        'aug_p': trial.suggest_categorical('aug_p', [0.1, 0.15, 0.20, 0.25, 0.3]),
        # 'p_dropout': trial.suggest_float('p_dropout', 0.01, 0.25, log=True),
    }

    # moe_args = {
    #     "input_dim": INPUT_DIM,
    #     "dim": params['hidden_dim'],
    #     "num_experts": 30,
    #     "num_slots": 10,
    #     "num_heads": 16,
    #     "slot_dim": 256,
    #     "keep_slots": True,  # if True, return the E*S aggregated features instead of the N transformed patch features
    #     "share_lora_weights": True,  # share the weights of the first low rank layer
    #     "dropout": 0.1,
    #     "auto_rank": True,  # automatically calculate the appropriate low rank for parameter efficiency
    # }
    #
    params['moe_args'] = None



    train_dataset = H5Dataset(
        csv_path=CSV_PATH_TRAIN,
        feats_path=H5_DIR_TRAIN,
        label_col=LABEL_COL,
        split='train',
        id_col=ID_COL,
        aug_prob=params['aug_p'],
        aug_feats_path=AUG_h5_DIR,
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
    EXP_NAME = input("Enter experiment name: ")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    optuna_df_path = f"./optuna_results/Architectural_Baselines/optuna_trials_{EXP_NAME}.csv"
    optuna_df = study.trials_dataframe()
    # remove the characters: user_attrs from the column names containing the user_attrs
    optuna_df.columns = [col.replace('user_attrs', '') for col in optuna_df.columns]
    optuna_df.to_csv(optuna_df_path, index=False)

    print("\n--- Optimization Finished ---")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Avg F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
