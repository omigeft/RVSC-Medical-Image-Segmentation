import argparse
import logging
import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from models import UNet
from models import UNetPlusPlus
from models import U2Net
from utils.data_loading import BasicDataset

def get_args():
    parser = argparse.ArgumentParser(description='Test the model on images and target masks')
    parser.add_argument('--pth', '-p', type=str, default='MODEL.pth', metavar='FILE',
                        help='Load model from a .pth file')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Directory of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory of ouput masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--channels', '-ch', type=int, default=1, help='Number of channels in input images')
    parser.add_argument('--classes', '-cl', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
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

    logging.info(f'Loading model from {args.pth}')
    state_dict = torch.load(args.pth, map_location=device)
    model_name = state_dict.pop('model_name', None)
    mask_values = state_dict.pop('mask_values', [0, 1])

    if model_name == 'unet++':
        net = UNetPlusPlus(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif model_name == 'u2net':
        net = U2Net(n_channels=args.channels, n_classes=args.classes)
    elif model_name == 'unet_cs':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=True)
    elif model_name == 'unet_c':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=True, s_attention=False)
    elif model_name == 'unet_s':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=True)
    elif model_name == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear,
                     c_attention=False, s_attention=False)
    else:
        raise ValueError(f'Model {model_name} not recognized')

    net.to(device=device)
    net.load_state_dict(state_dict)
    logging.info(f'Model {model_name} loaded!')

    test_dataset = BasicDataset(args.input, args.output, args.scale)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    dice, iou, prec, acc = test_net(net=net, device=device, test_loader=test_loader)
    logging.info(f'Dice score: {dice}')
    logging.info(f'Iou score: {iou}')
    logging.info(f'Precision: {prec}')
    logging.info(f'Accuracy: {acc}')
