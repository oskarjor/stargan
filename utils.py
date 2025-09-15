import torch


def get_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """Calculate TP, TN, FP, FN between predicted and ground truth binary tensors.

    Args:
        predictions: Binary tensor of model predictions
        labels: Binary tensor of ground truth labels

    Returns:
        tuple: (true_positives, true_negatives, false_positives, false_negatives)
    """
    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    true_negatives = ((predictions == 0) & (labels == 0)).sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).sum().item()

    return true_positives, true_negatives, false_positives, false_negatives
