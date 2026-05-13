import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import Namespace
import shutil
from scipy.ndimage import binary_dilation

from evaluate import evaluate
from unet import UNet
from unet import standard_UNet
# from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss, dice_for_class
from utils.data_loading import RasterPatchDataset
from utils.data_loading import match_pairs


def train_model(
        model,
        device,
        start_epoch: int = 1,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        channels_count: int = 9,
        patch_size: int = 256,
        stride: int = 256,
        pad_if_needed: bool = False,
        num_workers: int = 4,

        dir_img: str = None,
        dir_mask: str = None, 
        dir_checkpoint: str = 'UNet/checkpoints',
        dir_tensorboard: str = 'UNet/unet_experiment',

        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,

):


    # 1. Create dataset
    dataset = RasterPatchDataset(
        dir_img, dir_mask, \
        bands = list(range(1,(channels_count + 1))), \
        patch_size=patch_size, \
        stride=stride,\
        pad_if_needed = pad_if_needed, \
        check_imgMask_pair=True \
        )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=dir_tensorboard)
    # 记录超参数
    writer.add_text('config', 
        f"epochs: {epochs}, batch_size: {batch_size}, learning_rate: {learning_rate}, "
        f"val_percent: {val_percent}, save_checkpoint: {save_checkpoint}, amp: {amp}"
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = 0.5, min_lr = 1e-6, patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler("cuda",enabled=amp)
    criterion = nn.CrossEntropyLoss(ignore_index = 0) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    global_step = 0
    # 5. Begin training
    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch - start_epoch + 1}/{epochs}', unit='img') as pbar:
            for images, true_masks in train_loader:
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        dice_cls2 = dice_for_class(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            class_idx=2  # settlement class index
                        )
                        loss += 1 - dice_cls2

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), epoch)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # 每次epoch结束后记录一次权重和梯度的直方图
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                writer.add_histogram('Weights/' + tag, value.data.cpu(), epoch)
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), epoch)
        # 每次epoch结束后evaluate模型一次
        dice_score, oa_score, iou_score = evaluate(model, val_loader, device, amp)
        scheduler.step(dice_score)

        logging.info('Validation Dice score: {}'.format(dice_score))
        logging.info('Validation overall accuracy: {}'.format(oa_score))
        logging.info('Validation settlement iou: {}'.format(iou_score))
        try:
            # 记录验证分数
            writer.add_scalar('Dice/val', dice_score, epoch)
            writer.add_scalar('OA/val', oa_score, epoch)
            writer.add_scalar('iou/val', iou_score, epoch)
            # # 记录图片
            # img_grid = images[0].cpu()
            # mask_true = true_masks[0].float().cpu()
            # mask_pred = masks_pred.argmax(dim=1)[0].float().cpu()
        except:
            pass

        if save_checkpoint:
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            # state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(model.state_dict(), str(Path(dir_checkpoint) / 'checkpoint_epoch{:03d}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=1., help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=11, help='Number of classes')
    parser.add_argument('--channels', '-chn', type=int, default=9, help='Number of channels in input images')
    parser.add_argument('--patch_size', '-ps', type=int, default=256, help='Number of pixels in one side of the patch')
    parser.add_argument('--stride', type=int, default=256, help='Stride for patch extraction')
    parser.add_argument("pad_if_needed", action='store_true', default=False,
                        help='Pad the images if they are not divisible by the patch size')

    return parser.parse_args()


if __name__ == '__main__':
    # ====== VSCode调试用参数设置 ======
    config = Namespace(
        #
        config_infoNote = 'use CSMdataset train Unet\n',
        dir_img = r'my_datasets\2_datasets\4-reflectance', # 影像数据目录
        dir_mask = r'my_datasets\2_datasets\4-mask', # 标签数据目录
        dir_checkpoint = r'UNe\checkpoints\China_30m_CSMdataset_checkpoints', # model保存目录
        dir_tensorboard = r'UNet\unet_experiment\China_30m_CSMdataset_experiments', # TensorBoard日志目录

        load_model = False, # 是否加载预训练模型
        start_epoch = 1, # start from 1

        channels_count = 18, # 特征数量 - 数据加载
        classes = 3, # 类别数
        epochs = 150, # 训练轮数 - 训练过程
        batch_size = 8, # 批大小 - 训练过程
        num_workers = 4, # 数据加载时的工作线程
        patch_size = 256, # 每个patch的大小 - 数据加载
        stride = 256, # 滑动窗口的步长 - 数据加载
        pad_if_needed = True, # 是否在patch不满足大小时进行补零 - 数据加载
        dataset_split = np.array([9,1,0]), # Train, Validation, Test

        # hyperparameters
        learning_rate = 1e-3, # 学习率 - 训练过程
        momentum = 0.999, # 动量 - 优化器
        weight_decay = 1e-8, # 权重衰减 - 优化器
        gradient_clipping = 1.0, # 梯度裁剪 - 优化器
        amp = False, # 是否使用混合精度训练 - 训练和验证过程
        
        
    )
    # print(config.dir_img)

    # 保存config参数到config_info.txt
    config_path = os.path.join(config.dir_checkpoint, "config_info.txt")
    os.makedirs(config.dir_checkpoint, exist_ok=True)
    with open(config_path, "a") as f:
        f.write(config.config_infoNote)
        for k, v in vars(config).items():
            f.write(f"{k}: {v}\n")

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 0. Divide dataset into train validation, and test
    if config.dataset_split[2] != 0:
        items = match_pairs(config.dir_img, config.dir_mask)
        test_percent = config.dataset_split[2] / np.sum(config.dataset_split)
        test_count = int(np.round(len(items[0]) * test_percent))

        # 随机选择 test_count 个索引
        np.random.seed(1)
        all_indices = np.arange(len(items[0]))
        test_indices = np.random.choice(all_indices, size=test_count, replace=False)

        # 新建 test 文件夹
        test_mask_dir = os.path.join(os.path.dirname(os.path.dirname(config.dir_img)), "test_" + os.path.basename(config.dir_mask[:-1]))
        test_reflectance_dir = os.path.join(os.path.dirname(os.path.dirname(config.dir_img)), "test_" + os.path.basename(config.dir_img[:-1]))
        os.makedirs(test_mask_dir, exist_ok=True)
        os.makedirs(test_reflectance_dir, exist_ok=True)

        # # 移动文件
        for idx in test_indices:
            mask_file = items[1][idx]
            img_file = items[0][idx]
            shutil.move(mask_file, os.path.join(test_mask_dir, os.path.basename(mask_file)))
            shutil.copy(img_file, os.path.join(test_reflectance_dir, os.path.basename(img_file)))
        config.val_percent = config.dataset_split[1] / np.sum(config.dataset_split)
        logging.info(f'splited {test_count} of {len(items[0])} files into test set')
    else:
        config.val_percent = config.dataset_split[1] / np.sum(config.dataset_split)
        logging.info(f'Do not split dataset into training and testing')
        
        

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = standard_UNet(n_channels = config.channels_count, 
                                   n_classes = config.classes)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 # f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    if config.load_model:
        model_path = str(Path(config.dir_checkpoint) / 'checkpoint_epoch{:03d}.pth'.format(config.start_epoch - 1))
        state_dict = torch.load(model_path, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {model_path}')
    
    
    # model = torch.nn.DataParallel(model)
    model.to(device=device)

    allowed_keys = train_model.__code__.co_varnames
    filtered_config = {k: v for k, v in vars(config).items() if k in allowed_keys}

    try:
        train_model(model=model, device=device, **filtered_config)
        
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model=model, device=device, **filtered_config)

