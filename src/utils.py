import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, accuracy_score


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
    def __init__(self, patience=10, delta=0):
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


class CombinedCostLoss(nn.Module):
    def __init__(self, cost_matrix, alpha=1.0, beta=0.5):
        """
        Args:
            cost_matrix: The penalty matrix for misclassifications.
            alpha: Weight for the standard Cross-Entropy loss.
            beta: Weight for the Cost-Sensitive loss.
        """
        super(CombinedCostLoss, self).__init__()
        self.cost_matrix = cost_matrix
        self.alpha = alpha
        self.beta = beta

        # Standard CE loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # 1. Standard Cross-Entropy Loss
        if isinstance(logits, tuple):
            logits = logits[0]['logits']

        # 2. Cost-Sensitive Loss
        ce = self.ce_loss(logits, targets)
        probs = F.softmax(logits, dim=1)

        batch_costs = self.cost_matrix[targets]
        expected_costs = torch.sum(probs * batch_costs, dim=1)
        cs = expected_costs.mean()

        # 3. Combine them
        total_loss = (self.alpha * ce) + (self.beta * cs)
        return total_loss


def calculate_accuracies(predictions, ground_truths):
    """
    Calculate accuracy for each fold individually.

    Args:
        predictions (np.array): 2D array of shape (num_folds, num_samples) containing predicted labels for each fold.
        ground_truths (np.array): 1D array of shape (num_samples,) containing the true labels.
    """
    # Calculate accuracy for each fold individually

    fold_accuracies = []
    for i in range(predictions.shape[0]):
        fold_pred = predictions[i, :]
        # Compare fold predictions to ground truth
        acc = np.mean(fold_pred == ground_truths)
        fold_accuracies.append(acc)

    return fold_accuracies


class EMDCombinedLoss(nn.Module):
    """
    Cross-entropy + squared Earth Mover's Distance (Wasserstein) loss for the
    ordinal colorectal adenoma-carcinoma sequence.

    The EMD term replaces your hand-tuned cost matrix: each prediction is
    penalized by how far its probability mass sits from the true grade ALONG
    THE SEVERITY AXIS, so distant confusions cost more automatically.

    CRITICAL: your dataloader labels are NOT in severity order:
        0 = low-grade dysplasia
        1 = high-grade dysplasia
        2 = adenocarcinoma
        3 = benign          <-- least severe, but highest index

    Ascending severity:  benign < LGD < HGD < adenocarcinoma
    i.e. original labels: 3     <  0  <  1  <  2

    We reorder every prob/target vector by ORDINAL_ORDER before the CDF.
    """

    # original labels sorted by ASCENDING severity (position 0 = least severe)
    ORDINAL_ORDER = [3, 0, 1, 2]   # benign, LGD, HGD, adenocarcinoma

    def __init__(self, alpha=1.0, beta=1.0, under_grade_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # >1.0 penalizes UNDER-grading (predicting too benign) more than
        # over-grading -- your old FN>FP intent, now scaled by ordinal distance.
        # 1.0 = symmetric EMD.
        self.under_grade_weight = under_grade_weight

        self.standard_ce = nn.CrossEntropyLoss()
        # buffer so it moves with .to(device)
        self.register_buffer(
            "ordinal_order", torch.tensor(self.ORDINAL_ORDER, dtype=torch.long)
        )

    def forward(self, logits, hard_targets, soft_targets=None):
        num_classes = logits.size(1)

        # 1. Cross-entropy (soft in training, hard in val) -- unchanged
        if soft_targets is not None:
            ce_loss = F.cross_entropy(logits, soft_targets)
        else:
            ce_loss = self.standard_ce(logits, hard_targets)

        # 2. Squared EMD, computed in ordinal (severity-sorted) space
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(hard_targets, num_classes).float()

        # reorder columns so column index == severity position
        probs_ord = probs.index_select(1, self.ordinal_order)
        target_ord = target_onehot.index_select(1, self.ordinal_order)

        # 1-D EMD = L2 distance between CDFs
        cdf_pred = torch.cumsum(probs_ord, dim=1)
        cdf_true = torch.cumsum(target_ord, dim=1)
        cdf_diff = cdf_pred - cdf_true

        # cdf_diff > 0  <=>  predicted mass sits BELOW truth == under-grading
        weight = torch.ones_like(cdf_diff)
        weight[cdf_diff > 0] = self.under_grade_weight

        emd_loss = torch.sum(weight * cdf_diff.pow(2), dim=1).mean()

        # 3. Combine
        return (self.alpha * ce_loss) + (self.beta * emd_loss)


class LSCombinedCostLoss(nn.Module):
    def __init__(self, cost_matrix, alpha=1.0, beta=0.1):
        super(LSCombinedCostLoss, self).__init__()
        self.cost_matrix = cost_matrix
        self.alpha = alpha
        self.beta = beta
        # Initialize standard CE for validation fallback
        self.standard_ce = nn.CrossEntropyLoss()

    def forward(self, logits, hard_targets, soft_targets=None):
        # 1. Standard Cross-Entropy
        if soft_targets is not None:
            # Used during TRAINING with dynamic CPLS targets
            ce_loss = F.cross_entropy(logits, soft_targets)
        else:
            # Used during VALIDATION/TESTING with hard clinical targets
            ce_loss = self.standard_ce(logits, hard_targets)

        # 2. Cost-Sensitive Loss (Always anchors to hard clinical truth)
        probs = F.softmax(logits, dim=1)
        batch_costs = self.cost_matrix[hard_targets]
        expected_costs = torch.sum(probs * batch_costs, dim=1)
        cs_loss = expected_costs.mean()

        # 3. Combine
        total_loss = (self.alpha * ce_loss) + (self.beta * cs_loss)
        return total_loss



def initialize_uniform_smoothing(num_classes: int, gamma: float = 0.1) -> torch.Tensor:
    """
    Creates a standard label smoothing matrix.
    Returns a [num_classes, num_classes] tensor where row i is the soft target for true class i.
    """
    # Calculate the uniform penalty mass for incorrect classes
    smooth_val = gamma / (num_classes - 1)

    # Initialize a matrix filled with the smoothing value
    smoothing_matrix = torch.full((num_classes, num_classes), smooth_val)

    # Override the diagonal (the true classes) with the primary confidence mass
    smoothing_matrix.fill_diagonal_(1.0 - gamma)

    return smoothing_matrix


def compute_cpls_matrix(confusion_matrix: torch.Tensor, num_classes: int = 4, alpha: float = 0.1) -> torch.Tensor:
    """
    Updates the soft target smoothing matrix based on empirical confusion from the previous epoch.
    """
    new_smoothing_matrix = torch.zeros((num_classes, num_classes))

    for i in range(num_classes):
        # 1. Isolate the errors for true class i
        row_errors = confusion_matrix[i].clone()
        row_errors[i] = 0.0  # Zero out the correct predictions, we only care about mistakes

        total_errors = row_errors.sum()

        # 2. Calculate distribution weights for the alpha mass
        if total_errors > 0:
            # Distribute alpha proportionally to how often this specific mistake was made
            error_distribution = row_errors / total_errors
        else:
            # Fallback: If the model got 100% accuracy on this class, revert to uniform smoothing
            error_distribution = torch.full((num_classes,), 1.0 / (num_classes - 1))
            error_distribution[i] = 0.0

        # 3. Construct the new soft target row
        new_smoothing_matrix[i] = error_distribution * alpha
        new_smoothing_matrix[i, i] = 1.0 - alpha

    return new_smoothing_matrix


def update_confusion_matrix(conf_matrix: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor):
    """
    Helper function to tally predictions during the training loop.
    Call this inside your batch loop: update_confusion_matrix(epoch_cm, hard_targets, preds)
    """
    for t, p in zip(targets.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1


def create_cost_sensitive_smoothing_matrix(cost_matrix, alpha=0.1):
    """
    Converts a penalty cost matrix into a probability distribution for label smoothing.
    Distributes the `alpha` mass inversely proportional to the cost.
    """
    # 1. Ensure it's a tensor and grab its device (CPU or CUDA)
    if not isinstance(cost_matrix, torch.Tensor):
        cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)

    device = cost_matrix.device
    num_classes = len(cost_matrix)

    # Initialize the output matrix on the SAME device
    smoothing_matrix = torch.zeros((num_classes, num_classes), device=device)

    # Ensure alpha is a standard float to prevent device conflicts
    alpha = float(alpha)

    for i in range(num_classes):
        costs = cost_matrix[i].clone().detach().to(torch.float32)

        # 2. Create the mask on the SAME device
        mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        mask[i] = False
        incorrect_costs = costs[mask]

        # 3. Calculate the inverse of the costs
        inverses = 1.0 / (incorrect_costs + 1e-8)

        # 4. Normalize the inverses so they sum to 1.0
        normalized_weights = inverses / inverses.sum()

        # 5. Multiply by alpha to get the final distributed penalty mass
        alpha_distribution = normalized_weights * alpha

        # 6. Construct the row on the SAME device
        row = torch.zeros(num_classes, device=device)
        row[mask] = alpha_distribution
        row[i] = 1.0 - alpha

        smoothing_matrix[i] = row

    return smoothing_matrix



def get_folds_metrics(results_dict: dict):
    b_accs = [balanced_accuracy_score(results_dict.get('truths')[i], results_dict.get('preds')[i]) for i in range(5)]
    accs = [accuracy_score(results_dict.get('truths')[i], results_dict.get('preds')[i]) for i in range(5)]
    qwk_scores = [
        cohen_kappa_score(
            results_dict.get('truths')[i],
            results_dict.get('preds')[i],
            weights='quadratic'
        )
        for i in range(5)
    ]

    aucs = results_dict.get('aucs', 0)
    f1s = results_dict.get('f1s', 0)

    results_summary = {
    'AUCs': aucs,
    'Balanced_Accuracies': b_accs,
    'Accuracies': accs,
    'F1s': f1s,
    'QWKs': qwk_scores
    }

    return results_summary


class CostAwareCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.standard_ce = nn.CrossEntropyLoss()

    def forward(self, logits, hard_targets, soft_targets=None):
        if soft_targets is not None:
            # Uses the cost-aware smoothing matrix during training
            return F.cross_entropy(logits, soft_targets)
        else:
            # Fallback for validation/testing
            return self.standard_ce(logits, hard_targets)


class UnimodalSoftLabeler(nn.Module):
    """
    Generates unimodal soft targets for the ordinal CRC sequence, replacing
    uniform label smoothing. Mass leaks from the true grade to its ORDINAL
    neighbours (by severity), decaying with distance -- so a benign label
    leaks toward LGD, never toward adenocarcinoma.

    SORD (Diaz & Marathe, CVPR 2019):
        soft(t)_j = softmax_j( -|rank_t - rank_j|^p / T )

    Dataloader labels are NOT in severity order:
        0=LGD, 1=HGD, 2=adenocarcinoma, 3=benign
    Ascending severity: benign(3) < LGD(0) < HGD(1) < adeno(2)
    """

    ORDINAL_ORDER = [3, 0, 1, 2]  # dataloader labels, ascending severity

    def __init__(self, num_classes=4, distance_power=2.0, temperature=1.0):
        super().__init__()
        order = torch.tensor(self.ORDINAL_ORDER, dtype=torch.long)
        ranks = torch.argsort(order).float()          # rank[label] -> [1,2,3,0]

        # dist[t, j] = |rank_t - rank_j| on the severity axis
        dist = (ranks.unsqueeze(1) - ranks.unsqueeze(0)).abs()
        table = F.softmax(-(dist ** distance_power) / temperature, dim=1)

        # row t = soft label for true class t, indexed in DATALOADER order
        self.register_buffer("table", table)

    def forward(self, hard_targets):
        table = self.table.to(hard_targets.device)    # device safety net
        return table[hard_targets]
