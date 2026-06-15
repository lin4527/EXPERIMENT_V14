"""
训练循环 — CPU友好轻量化方案
- 小batch size (2)
- 少量epoch (10-20)
- 早停机制
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from config import OUTPUT_DIR


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        B, C = probs.shape[:2]
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        if C > 1:
            dice_per_class = dice_per_class[1:]
        return 1.0 - dice_per_class.mean()


def compute_val_dice(model, val_loader, device):
    """Computes binary foreground Dice (not per-class) — more stable for small datasets"""
    model.eval()
    dices = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            # Binary foreground Dice: any non-background class counts as foreground
            pred_fg = (pred > 0).float()
            mask_fg = (masks > 0).float()
            inter = (pred_fg * mask_fg).sum()
            total = pred_fg.sum() + mask_fg.sum()
            if total > 0:
                dices.append((2.0 * inter / total).item())
    return float(np.mean(dices)) if dices else 0.0


def train_unet(model, dataset, val_dataset=None, epochs=10, batch_size=2,
               lr=1e-4, device=None, save_path=None, progress_callback=None):
    """
    训练 U-Net 模型 (CPU友好)
    :param model: UNet 实例
    :param dataset: 训练数据集
    :param val_dataset: 验证数据集 (可选)
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :param device: 'cpu' / 'cuda'
    :param save_path: 模型保存路径
    :param progress_callback: 可选回调 fn(epoch, loss, val_dice) 用于GUI更新
    :return: 训练历史 dict
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 必须在目标device上创建model再传进DataLoader
    model = model.to(device)

    # Windows CPU 下 pin_memory 必须为 False
    use_pin = (device == 'cuda')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=use_pin)

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=False)

    # 损失函数
    dice_loss = DiceLoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_dice': [], 'epoch_time': []}
    best_val_dice = 0.0
    patience_counter = 0
    patience = 5

    print(f"[Train] device={device}, epochs={epochs}, bs={batch_size}, samples={len(dataset)}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_losses = []
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_loss = float(np.mean(train_losses))
        history['train_loss'].append(avg_loss)

        # Validation
        val_dice = compute_val_dice(model, val_loader, device) if val_loader else 0.0
        history['val_dice'].append(val_dice)

        # Timing
        elapsed = time.time() - epoch_start
        history['epoch_time'].append(elapsed)

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | ValDice: {val_dice:.4f} | Time: {elapsed:.1f}s")

        # 回调 GUI
        if progress_callback:
            progress_callback(epoch + 1, avg_loss, val_dice)

        scheduler.step()

        # 早停 + 保存
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_dice,
                }, save_path)
                print(f"  [Save] best model -> {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    print(f"[Done] Best ValDice: {best_val_dice:.4f}")
    return history
