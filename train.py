import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from models import UNet
from models import UNetPlusPlus
from models import U2Net
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

dir_img = Path('../train_data/imgs/')
dir_mask = Path('../train_data/i-masks/')
dir_checkpoint = Path('../i-checkpoints/')

def train_model(
        model,
        device,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_percent: float,
        save_checkpoint: bool,
        img_scale: float,
        amp: bool,
        weight_decay: float,
        momentum: float,
        gradient_clipping: float,
        epochs_per_checkpoint: int,
        loss_function: str,
        optimizer_name: str,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Model:           {model.model_name}
        Channels:        {model.n_channels}
        Classes:         {model.n_classes}
        Bilinear:        {model.bilinear}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed precision: {amp}
        Weight decay:    {weight_decay}
        Momentum:        {momentum}
        Gradient clipping: {gradient_clipping}
        Epochs per checkpoint: {epochs_per_checkpoint}
        Loss function:   {loss_function}
        Optimizer:       {optimizer_name}
        
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    else:   # optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        if loss_function == 'dice':
                            loss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        elif loss_function == 'ce':
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        else:   # loss_function == 'dice+ce'
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        if loss_function == 'dice':
                            loss = dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                        elif loss_function == 'ce':
                            loss = criterion(masks_pred, true_masks)
                        else:   # loss_function == 'dice+ce'
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        dice_score, iou_score, prec_score, acc_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(dice_score)

                        logging.info('Validation Dice score: {}'.format(dice_score))
                        logging.info('Validation Iou score: {}'.format(iou_score))
                        logging.info('Validation Precision score: {}'.format(prec_score))
                        logging.info('Validation Accuracy score: {}'.format(acc_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': dice_score,
                                'validation Iou': iou_score,
                                'validation Precision': prec_score,
                                'validation Accuracy': acc_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint and epoch % epochs_per_checkpoint == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'{model.model_name}_checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

    wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-md', metavar='M', type=str, default='unet',
                        help='Name of model ("unet", "unet++", "u2net")')
    parser.add_argument('--channels', '-ch', type=int, default=1, help='Number of channels in input images')
    parser.add_argument('--classes', '-cl', type=int, default=2, help='Number of classes')
    parser.add_argument('--bilinear', '-bl', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-bs', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--weight-decay', '-w', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--momentum', '-mm', type=float, default=0.999, help='Momentum')
    parser.add_argument('--gradient-clipping', '-gc', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--epochs-per-checkpoint', '-epc', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--loss', '-ls', type=str, default='dice+ce', help='Loss function ("dice", "ce", "dice+ce")')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Optimizer ("adam", "rmsprop")')

    parsed_args = parser.parse_args()
    if parsed_args.model not in ['unet', 'unet++', 'u2net']:
        raise ValueError('Model must be one of "unet", "unet++", "u2net"')
    if parsed_args.loss not in ['dice', 'ce', 'dice+ce']:
        raise ValueError('Loss must be one of "dice", "ce", "dice+ce"')
    if parsed_args.optimizer not in ['adam', 'rmsprop']:
        raise ValueError('Optimizer must be one of "adam", "rmsprop"')

    return parsed_args


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model == 'unet++':
        model = UNetPlusPlus(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'u2net':
        model = U2Net(n_channels=args.channels, n_classes=args.classes)
    else:   # args.model == 'unet'
        model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=True,
            img_scale=args.scale,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            epochs_per_checkpoint=args.epochs_per_checkpoint,
            loss_function=args.loss,
            optimizer_name=args.optimizer
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            save_checkpoint=True,
            img_scale=args.scale,
            amp=args.amp,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            gradient_clipping=args.gradient_clipping,
            epochs_per_checkpoint=args.epochs_per_checkpoint,
            loss_function=args.loss,
            optimizer_name=args.optimizer
        )