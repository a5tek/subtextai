"""
Loss Function Library for Severity Distress Modeling (E1 - E5)

Implements:
  - E1: Standard Categorical Cross-Entropy Loss
  - E2: Class-Weighted Cross-Entropy Loss (with severe crisis boost factor)
  - E3: Multiclass Focal Loss (Lin et al., 2017)
  - E4: Class-Weighted Focal Loss
  - E5: Consistent Rank Logits (CORAL) Ordinal Loss (Cao et al., 2020)
  - Helper functions for automated class weighting and experiment factory instantiation.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[List[int], np.ndarray, torch.Tensor],
    num_classes: int = 4,
    method: str = "inverse_frequency",
    severe_boost: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Computes class weighting tensor from training target distribution.

    Args:
        labels: 1D collection of ground-truth integer labels [0, num_classes-1].
        num_classes: Total number of severity categories (default: 4).
        method: Weight calculation strategy ('inverse_frequency' or 'effective_num').
        severe_boost: Multiplicative multiplier applied specifically to class 3 (Severe Crisis).
        device: Target torch device.

    Returns:
        torch.FloatTensor of shape (num_classes,) normalized such that mean(weights) = 1.0.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels_arr = np.array(labels)
    else:
        labels_arr = labels

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = np.sum(labels_arr == c)

    # Guard against zero-division for empty classes in mini-batches/splits
    counts = np.maximum(counts, 1.0)
    total_samples = float(len(labels_arr)) if len(labels_arr) > 0 else float(num_classes)

    if method == "inverse_frequency":
        # w_c = N / (K * N_c)
        weights = total_samples / (num_classes * counts)
    elif method == "sqrt_inverse":
        weights = np.sqrt(total_samples / (num_classes * counts))
    else:
        weights = np.ones(num_classes, dtype=np.float64)

    # Apply severe class boost to class index 3 (Severe Crisis) if applicable
    severe_idx = num_classes - 1
    if severe_boost > 0 and severe_idx < num_classes:
        weights[severe_idx] *= float(severe_boost)

    # Normalize weights so that the mean equals 1.0 (maintains loss scale)
    weights = weights / np.mean(weights)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


class WeightedCrossEntropyLoss(nn.Module):
    """
    Class-Weighted Cross Entropy Loss for mitigating class imbalance and
    penalizing false negatives in critical distress severity classes.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            weight: 1D Tensor of shape (num_classes,) assigning weight to each class.
            reduction: Loss reduction ('mean', 'sum', 'none').
            label_smoothing: Float in [0.0, 1.0] for regularizing overconfident predictions.
        """
        super().__init__()
        self.register_buffer("weight", weight)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits of shape (B, num_classes).
            targets: Ground-truth integer labels of shape (B,).
        """
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=w,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss (Lin et al., ICCV 2017).
    
    Down-weights easy, well-classified examples to focus gradients on hard,
    subtle, and ambiguous distress examples:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        
    where p_t is the predicted probability for the ground-truth class.
    When gamma=0 and alpha=None, reduces identically to standard Cross Entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter (gamma >= 0). Higher values reduce gradient
                   contribution from easy samples.
            alpha: Optional class weights of shape (num_classes,) or None for unweighted.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"Focal loss gamma must be non-negative, got {gamma}")
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer("alpha", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss numerically stably using log_softmax.
        
        Args:
            logits: Model predictions of shape (B, num_classes).
            targets: Integer class indices of shape (B,).
            
        Returns:
            Scalar loss tensor (or tensor of shape (B,) if reduction='none').
        """
        # Log-probabilities: shape (B, K)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log p_t for the true target class: shape (B,)
        targets_expanded = targets.view(-1, 1)
        log_pt = log_probs.gather(dim=-1, index=targets_expanded).squeeze(-1)
        
        # True class probability p_t: shape (B,)
        pt = log_pt.exp()
        
        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - pt).clamp(min=0.0) ** self.gamma
        
        # Base focal loss
        loss = -modulating_factor * log_pt

        # Apply class-specific alpha weighting if provided
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            alpha_t = alpha_tensor.gather(dim=0, index=targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class OrdinalCoralLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) Loss for Ordinal Severity Classification (Cao et al., 2020).
    
    Transforms K-class ordinal classification into K-1 binary classification tasks:
        Task k: Does the distress severity exceed category k? (y > k for k in 0, ..., K-2)
    
    This preserves the natural hierarchy: Control < Low Stress < Moderate < Severe Crisis.
    Conflating Control with Severe incurs cumulative penalties across all cutoffs, unlike
    standard multiclass cross-entropy which treats all errors symmetrically.
    """

    def __init__(self, num_classes: int = 4, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.num_cutoffs = num_classes - 1
        self.reduction = reduction

    def label_to_coral_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Encodes integer ordinal rank [0, K-1] into (K-1) binary indicator vector.
        
        For class y in [0, K-1]:
            coral_targets[k] = 1 if y > k else 0, for k in [0, K-2]
            
        Example for K=4:
            y=0 (Control):   [0, 0, 0]
            y=1 (Low):       [1, 0, 0]
            y=2 (Moderate):  [1, 1, 0]
            y=3 (Severe):    [1, 1, 1]
        """
        B = targets.size(0)
        coral_targets = torch.zeros((B, self.num_cutoffs), dtype=torch.float32, device=targets.device)
        for k in range(self.num_cutoffs):
            coral_targets[:, k] = (targets > k).float()
        return coral_targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes BCE loss across the K-1 ordinal cutoffs.
        
        Args:
            logits: Tensor of shape (B, K-1) representing cumulative cutoff logits.
                    (If logits has shape (B, K), the first K-1 logits or differences are used).
            targets: Ground-truth integer labels of shape (B,).
        """
        # If model outputs K logits, slice the first K-1 for ordinal cutoff representation
        if logits.size(1) == self.num_classes:
            cutoff_logits = logits[:, :self.num_cutoffs]
        elif logits.size(1) == self.num_cutoffs:
            cutoff_logits = logits
        else:
            raise ValueError(
                f"Expected logits with {self.num_cutoffs} or {self.num_classes} columns, "
                f"got {logits.size(1)}"
            )

        binary_targets = self.label_to_coral_targets(targets)
        loss = F.binary_cross_entropy_with_logits(cutoff_logits, binary_targets, reduction=self.reduction)
        return loss

    @staticmethod
    def logits_to_ordinal_preds(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Converts (B, K-1) cutoff logits into integer predictions [0, K-1].
        
        A prediction is rank r if the model predicts P(y > k) >= threshold for all k < r.
        Equivalently, predicted rank = sum_{k=0}^{K-2} I(sigmoid(logits_k) >= threshold).
        """
        probs = torch.sigmoid(logits)
        exceeds = (probs >= threshold).long()
        predicted_rank = torch.sum(exceeds, dim=-1)
        return predicted_rank

    @staticmethod
    def logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, K-1) cutoff logits into valid probability distribution over K classes.
        
        P(y = 0) = 1 - P(y > 0)
        P(y = k) = P(y > k-1) - P(y > k) for 0 < k < K-1
        P(y = K-1) = P(y > K-2)
        """
        B, K_minus_1 = logits.size()
        K = K_minus_1 + 1
        cum_probs = torch.sigmoid(logits)  # P(y > k)
        
        # Enforce monotonicity if cutoffs deviate slightly:
        # P(y > 0) >= P(y > 1) >= ... >= P(y > K-2)
        class_probs = torch.zeros((B, K), dtype=torch.float32, device=logits.device)
        class_probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, K_minus_1):
            class_probs[:, k] = torch.clamp(cum_probs[:, k - 1] - cum_probs[:, k], min=0.0)
        class_probs[:, K - 1] = cum_probs[:, K_minus_1 - 1]
        
        # Normalize in case of small numerical drift
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return class_probs


def get_loss_function(
    config: Dict[str, Any],
    train_labels: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
    num_classes: int = 4,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Factory function to instantiate loss criteria for experiments E1 through E5.

    Args:
        config: Dictionary containing loss configuration (e.g., from loss_experiments.yaml).
                Expected structure: {'type': ..., 'gamma': ..., 'weights': ..., 'severe_boost_factor': ...}
        train_labels: Ground truth training labels for dynamic class weight calculation.
        num_classes: Number of target categories.
        device: Torch device to place weight buffers on.

    Returns:
        Instantiated nn.Module loss criterion.
    """
    loss_cfg = config.get("loss", config)
    loss_type = loss_cfg.get("type", "cross_entropy").lower()

    if loss_type in ["cross_entropy", "ce"]:
        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type in ["weighted_cross_entropy", "wce"]:
        severe_boost = float(loss_cfg.get("severe_boost_factor", 1.0))
        if train_labels is not None:
            weights = compute_class_weights(
                train_labels,
                num_classes=num_classes,
                method="inverse_frequency",
                severe_boost=severe_boost,
                device=device,
            )
        elif "weights" in loss_cfg and isinstance(loss_cfg["weights"], list):
            weights = torch.tensor(loss_cfg["weights"], dtype=torch.float32, device=device)
        else:
            # Default fallback balanced weights with severe boost applied
            default_weights = [1.0, 1.0, 1.5, 3.0 * severe_boost]
            weights = torch.tensor(default_weights, dtype=torch.float32, device=device)
            weights = weights / weights.mean()

        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        return WeightedCrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    elif loss_type in ["focal_loss", "fl"]:
        gamma = float(loss_cfg.get("gamma", 2.0))
        reduction = loss_cfg.get("reduction", "mean")
        return FocalLoss(gamma=gamma, alpha=None, reduction=reduction)

    elif loss_type in ["weighted_focal_loss", "wfl"]:
        gamma = float(loss_cfg.get("gamma", 2.0))
        reduction = loss_cfg.get("reduction", "mean")
        severe_boost = float(loss_cfg.get("severe_boost_factor", 1.0))

        if train_labels is not None:
            weights = compute_class_weights(
                train_labels,
                num_classes=num_classes,
                method="inverse_frequency",
                severe_boost=severe_boost,
                device=device,
            )
        elif "alpha" in loss_cfg and isinstance(loss_cfg["alpha"], list):
            weights = torch.tensor(loss_cfg["alpha"], dtype=torch.float32, device=device)
        else:
            default_weights = [1.0, 1.0, 1.5, 3.0 * severe_boost]
            weights = torch.tensor(default_weights, dtype=torch.float32, device=device)
            weights = weights / weights.mean()

        return FocalLoss(gamma=gamma, alpha=weights, reduction=reduction)

    elif loss_type in ["ordinal_coral", "coral"]:
        reduction = loss_cfg.get("reduction", "mean")
        return OrdinalCoralLoss(num_classes=num_classes, reduction=reduction)

    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. Supported: "
            f"['cross_entropy', 'weighted_cross_entropy', 'focal_loss', 'weighted_focal_loss', 'ordinal_coral']"
        )
