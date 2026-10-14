"""
PyTorch 工程基础：Checkpoint + 学习率调度 + 早停
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FiveLayerNet


def get_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    val_set   = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, best_val_loss, path='checkpoint.pth'):
    """
    保存训练状态，支持断点续训。

    必须同时保存 optimizer.state_dict()：Adam 内部维护 m_t / v_t
    两个指数移动平均，若不保存，续训时优化器从零起步，前几轮会剧烈震荡。
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    """恢复训练状态，返回 (start_epoch, best_val_loss)；无存档则从头开始。"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_val_loss']
    return 0, float('inf')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, val_loader = get_loaders(batch_size=128)
    model     = FiveLayerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 验证 loss 连续 3 个 epoch 没改善，lr 乘以 0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    patience   = 5
    no_improve = 0
    max_epochs = 20

    start_epoch, best_val_loss = load_checkpoint(model, optimizer)
    if start_epoch > 0:
        print(f"从 checkpoint 恢复：已完成 {start_epoch} 轮，历史最佳 val_loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{max_epochs} | "
              f"Train {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val {val_loss:.4f}/{val_acc:.1f}% | "
              f"LR {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss)
            print(f"  → 保存 checkpoint（val_loss 改善到 {best_val_loss:.4f}）")
        else:
            no_improve += 1
            print(f"  → 无改善 {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"Early stopping 触发，在 epoch {epoch+1} 停止训练")
                break

    print(f"\n训练结束，最佳 val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
