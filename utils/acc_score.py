import torch
from torch import Tensor


def accuracy(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Calculate Accuracy
    # True Positives (TP) + True Negatives (TN)
    correct_predictions = ((input == target) * (target == 1)).sum()

    # Total number of predictions
    total_predictions = torch.numel(input)

    accuracy = correct_predictions / (total_predictions + epsilon)
    return accuracy