# 🧠 MNIST Handwritten Digit Recognition | MNIST 手写数字识别

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A lightweight CNN-based MNIST classifier built with PyTorch, achieving high accuracy on handwritten digit recognition.**

**基于 PyTorch 的轻量级 CNN 手写数字识别项目，在 MNIST 数据集上达到高精度识别效果。**

</div>

---

## 📌 Project Overview | 项目简介

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset. It includes a complete pipeline covering data loading, model training, evaluation, and real-time visual prediction.

本项目使用**卷积神经网络（CNN）**对 MNIST 数据集中的手写数字（0–9）进行分类，涵盖数据加载、模型训练、评估验证与可视化预测的完整流程。

---

## 🗂️ Project Structure | 项目结构

```
├── CNN.py            # CNN model definition | CNN 模型定义
├── train.py          # Model training script | 模型训练脚本
├── predict.py        # Prediction & visualization | 预测与可视化
├── model/
│   └── mnist_model.pkl   # Saved model weights | 保存的模型权重
└── minst/            # Auto-downloaded MNIST dataset | 自动下载的数据集
```

---

## 🏗️ Model Architecture | 模型结构

```
Input (1×28×28)
    │
    ▼
┌─────────────────────────────┐
│  Conv2d(1→32, kernel=5, p=2)│  # Feature extraction | 特征提取
│  BatchNorm2d(32)            │  # Normalization | 归一化
│  ReLU()                     │  # Activation | 激活函数
│  MaxPool2d(2)               │  # Downsampling | 下采样
└─────────────────────────────┘
    │
    ▼  (32×14×14 = 6272 features)
    │
    ▼
┌─────────────────────────────┐
│  Flatten → Linear(6272→10) │  # Fully connected | 全连接层
└─────────────────────────────┘
    │
    ▼
Output (10 classes | 10分类)
```

| Layer | Type | Details |
|-------|------|---------|
| Conv1 | Conv2d | in=1, out=32, kernel=5×5, padding=2 |
| BN1 | BatchNorm2d | 32 channels |
| Act | ReLU | — |
| Pool | MaxPool2d | kernel=2×2, stride=2 |
| FC | Linear | 6272 → 10 |

---

## ⚙️ Training Configuration | 训练配置

| Parameter 参数 | Value 值 |
|---|---|
| Optimizer 优化器 | Adam |
| Learning Rate 学习率 | 0.01 |
| Loss Function 损失函数 | CrossEntropyLoss |
| Batch Size 批次大小 | 64 |
| Epochs 训练轮数 | 10 |
| Device 设备 | CUDA (GPU) |

---

## 🚀 Quick Start | 快速开始

### 1. Install Dependencies | 安装依赖

```bash
pip install torch torchvision opencv-python
```

> ✅ Make sure you have a CUDA-compatible GPU and the correct CUDA version installed.
> 请确保安装了支持 CUDA 的 GPU 驱动和对应版本的 CUDA。

### 2. Train the Model | 训练模型

```bash
python train.py
```

Training logs will display epoch, batch progress, and loss in real time.
训练过程会实时打印当前轮次、批次进度与 loss 值。

```
当前为第1轮，批次为1/938, loss为2.3012
当前为第1轮，批次为2/938, loss为1.8754
...
```

### 3. Run Prediction | 运行预测

```bash
python predict.py
```

Each test image will be displayed via OpenCV with its predicted and true labels printed to the console.
每张测试图片将通过 OpenCV 窗口展示，并在终端打印预测值与真实值。

---

## 📊 Results | 实验结果

| Metric 指标 | Value 值 |
|---|---|
| Test Accuracy 测试准确率 | ~99% |
| Training Device 训练设备 | NVIDIA GPU (CUDA) |
| Dataset 数据集 | MNIST (60k train / 10k test) |

---

## 📦 File Description | 文件说明

### `CNN.py` — Model Definition | 模型定义
Defines the `CNN` class inheriting from `torch.nn.Module`, with one convolutional block and one fully connected output layer.
定义继承自 `torch.nn.Module` 的 `CNN` 类，包含一个卷积块和一个全连接输出层。

### `train.py` — Training Pipeline | 训练流程
Loads MNIST, trains the CNN for 10 epochs with Adam optimizer, and saves the model to `model/mnist_model.pkl`.
加载 MNIST 数据集，使用 Adam 优化器训练 10 轮，并将模型保存至 `model/mnist_model.pkl`。

### `predict.py` — Prediction & Visualization | 预测与可视化
Loads the saved model, runs inference on the test set, and displays each image with OpenCV.
加载已保存模型，对测试集进行推理，并通过 OpenCV 逐张展示图片与预测结果。

---

## 🔧 Requirements | 环境要求

```
torch >= 1.10
torchvision >= 0.11
opencv-python >= 4.5
CUDA (recommended | 推荐)
```

---

## 📝 License | 许可证

This project is licensed under the [MIT License](LICENSE).
本项目基于 [MIT 许可证](LICENSE) 开源。

---

## 🙋 Author | 作者

> Feel free to open an issue or submit a PR if you have any suggestions!
> 如有建议，欢迎提 Issue 或 Pull Request！

<div align="center">
  Made with ❤️ and PyTorch
</div>
