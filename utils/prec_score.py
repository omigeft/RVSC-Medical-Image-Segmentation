import torch
from torch import Tensor


def precision(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Calculate Precision
    # True Positives (TP)
    true_positives = (input * target).sum()

    # Predicted Positives
    predicted_positives = input.sum()

    precision = true_positives / (predicted_positives + epsilon)
    return precision