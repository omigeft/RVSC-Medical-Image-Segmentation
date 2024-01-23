import torch
from torch import Tensor


def iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    intersection = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - intersection
    union = torch.clamp(union, min=epsilon)  # Prevent division by zero

    iou = intersection / union
    return iou.mean()


def multiclass_iou_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU for all classes
    return iou_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def iou_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # IoU loss (objective to minimize) between 0 and 1
    fn = multiclass_iou_coeff if multiclass else iou_coeff
    return 1 - fn(input, target, reduce_batch_first=True)