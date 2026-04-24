# 🧠 MNIST 手写数字识别

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**基于 PyTorch 的轻量级 CNN 手写数字识别项目，在 MNIST 数据集上达到高精度识别效果。**

</div>

---

## 📌 项目简介

本项目使用**卷积神经网络（CNN）**对 MNIST 数据集中的手写数字（0–9）进行分类，涵盖数据加载、模型训练、评估验证与可视化预测的完整流程。

---

## 🗂️ 项目结构

```
├── CNN.py                    # CNN 模型定义
├── train.py                  # 模型训练脚本
├── predict.py                # 预测与可视化
├── model/
│   └── mnist_model.pkl       # 保存的模型权重
└── minst/                    # 自动下载的 MNIST 数据集
```

---

## 🏗️ 模型结构

```
Input (1×28×28)
    │
    ▼
┌─────────────────────────────┐
│  Conv2d(1→32, kernel=5, p=2)│  # 特征提取
│  BatchNorm2d(32)            │  # 归一化
│  ReLU()                     │  # 激活函数
│  MaxPool2d(2)               │  # 下采样
└─────────────────────────────┘
    │
    ▼  (32×14×14 = 6272 features)
    │
    ▼
┌─────────────────────────────┐
│  Flatten → Linear(6272→10) │  # 全连接层
└─────────────────────────────┘
    │
    ▼
Output（10 分类）
```

| 层名 | 类型 | 参数 |
|------|------|------|
| Conv1 | Conv2d | in=1, out=32, kernel=5×5, padding=2 |
| BN1 | BatchNorm2d | 32 channels |
| Act | ReLU | — |
| Pool | MaxPool2d | kernel=2×2, stride=2 |
| FC | Linear | 6272 → 10 |

---

## ⚙️ 训练配置

| 参数 | 值 |
|------|----|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 10 |
| 训练设备 | CUDA (GPU) |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision opencv-python
```

> ✅ 请确保安装了支持 CUDA 的 GPU 驱动和对应版本的 CUDA。

### 2. 训练模型

```bash
python train.py
```

训练过程会实时打印当前轮次、批次进度与 loss 值：

```
当前为第1轮，批次为1/938, loss为2.3012
当前为第1轮，批次为2/938, loss为1.8754
...
```

### 3. 运行预测

```bash
python predict.py
```

每张测试图片将通过 OpenCV 窗口展示，并在终端打印预测值与真实值。

---

## 📊 实验结果

| 指标 | 值 |
|------|----|
| 测试准确率 | ~99% |
| 训练设备 | NVIDIA GPU (CUDA) |
| 数据集 | MNIST（60k 训练 / 10k 测试）|

---

## 📦 文件说明

### `CNN.py` — 模型定义
定义继承自 `torch.nn.Module` 的 `CNN` 类，包含一个卷积块和一个全连接输出层。

### `train.py` — 训练流程
加载 MNIST 数据集，使用 Adam 优化器训练 10 轮，并将模型保存至 `model/mnist_model.pkl`。

### `predict.py` — 预测与可视化
加载已保存模型，对测试集进行推理，并通过 OpenCV 逐张展示图片与预测结果。

---

## 🔧 环境要求

```
torch >= 1.10
torchvision >= 0.11
opencv-python >= 4.5
CUDA（推荐）
```

---

## 📝 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

---

## 🙋 作者

> 如有建议，欢迎提 Issue 或 Pull Request！

<div align="center">
  Made with ❤️ and PyTorch
</div>
