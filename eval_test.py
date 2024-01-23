import argparse
import logging
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset

def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Directory of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory of ouput masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


def test_net(net, device, test_loader, amp=False):
    net.eval()
    dice_score, iou_score, prec_score, acc_score = evaluate(net, test_loader, device, amp)
    return dice_score, iou_score, prec_score, acc_score


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device=device)

    logging.info(f'Loading model {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    test_dataset = BasicDataset(args.input, args.output, args.scale)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    dice, iou, prec, acc = test_net(net=net, device=device, test_loader=test_loader)
    logging.info(f'Dice score: {dice}')
    logging.info(f'Iou score: {iou}')
    logging.info(f'Precision: {prec}')
    logging.info(f'Accuracy: {acc}')
