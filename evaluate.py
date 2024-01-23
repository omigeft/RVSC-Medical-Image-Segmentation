import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import multiclass_iou_coeff, iou_coeff
from utils.prec_score import precision
from utils.acc_score import accuracy


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    prec_score = 0
    acc_score = 0

    # iterate over the validation set
    with torch.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type`
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # compute the Iou score
                iou_score += iou_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # compute the Precision score
                prec_score += precision(mask_pred, mask_true)
                # compute the Accuracy score
                acc_score += accuracy(mask_pred, mask_true)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                # compute the Iou score, ignoring background
                iou_score += multiclass_iou_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                # compute the Precision score, ignoring background
                prec_score += precision(mask_pred[:, 1:], mask_true[:, 1:])
                # compute the Accuracy score, ignoring background
                acc_score += accuracy(mask_pred[:, 1:], mask_true[:, 1:])

    net.train()
    return dice_score / max(num_val_batches, 1), iou_score / max(num_val_batches, 1), \
        prec_score / max(num_val_batches, 1), acc_score / max(num_val_batches, 1)
