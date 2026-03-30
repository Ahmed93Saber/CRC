import torch

# Define your matrices here as standard Python dictionaries/lists
COST_MATRICES = {
    "baseline": [
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 3.0, 10.0],
        [0.0, 3.0, 0.0, 5.0],
        [2.0, 10.0, 5.0, 0.0]
    ],
    "asymmetric_risk": [
        [0.0, 1.0, 3.0, 8.0],
        [3.0, 0.0, 2.0, 6.0],
        [8.0, 4.0, 0.0, 4.0],
        [20.0, 15.0, 5.0, 0.0]
    ],
    "squared_distance": [
        [0.0, 1.0, 4.0, 9.0],
        [1.0, 0.0, 1.0, 4.0],
        [4.0, 1.0, 0.0, 1.0],
        [9.0, 4.0, 1.0, 0.0]
    ],
    "malignant_bottleneck": [
        [0.0, 1.0, 1.0, 10.0],
        [1.0, 0.0, 1.0, 10.0],
        [2.0, 1.0, 0.0, 5.0],
        [15.0, 15.0, 8.0, 0.0]
    ]
}


def get_cost_matrix(name: str, device: torch.device) -> torch.Tensor:
    """Fetches the specified cost matrix and initializes it as a Tensor on the target device."""
    if name not in COST_MATRICES:
        raise ValueError(f"Cost matrix '{name}' not found. Available matrices: {list(COST_MATRICES.keys())}")

    return torch.tensor(COST_MATRICES[name], dtype=torch.float32, device=device)