"""
训练脚本：对比三种初始化方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json

from model import FiveLayerNet
from init_methods import INIT_METHODS


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_with_init_method(init_method_name, epochs=10, lr=0.01, batch_size=128):
    """
    使用指定初始化方法训练模型

    Args:
        init_method_name: 初始化方法名称 ('default', 'xavier', 'kaiming')
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小

    Returns:
        history: 训练历史 (包含每个epoch的loss和accuracy)
    """
    print(f"\n{'='*60}")
    print(f"开始训练：{init_method_name} 初始化")
    print(f"{'='*60}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = FiveLayerNet().to(device)

    # 应用初始化方法
    init_func = INIT_METHODS[init_method_name]
    init_func(model)

    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # 训练循环
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 打印
        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    elapsed_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {elapsed_time:.2f}秒")
    print(f"最终测试准确率: {history['test_acc'][-1]:.2f}%")

    return history


def main():
    """主函数：依次训练三种初始化方法"""
    epochs = 10
    lr = 0.01
    batch_size = 128

    results = {}

    # 训练三种初始化方法
    for method_name in ['default', 'xavier', 'kaiming']:
        history = train_with_init_method(method_name, epochs=epochs, lr=lr, batch_size=batch_size)
        results[method_name] = history

    # 保存结果
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n实验结果已保存到 results.json")

    # 打印对比总结
    print(f"\n{'='*60}")
    print("实验总结")
    print(f"{'='*60}")
    print(f"{'初始化方法':<15} {'最终测试准确率':<20} {'第1轮测试准确率':<20}")
    print(f"{'-'*60}")
    for method_name, history in results.items():
        final_acc = history['test_acc'][-1]
        first_acc = history['test_acc'][0]
        print(f"{method_name:<15} {final_acc:>18.2f}% {first_acc:>18.2f}%")


if __name__ == '__main__':
    main()
