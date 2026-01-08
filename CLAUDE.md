# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于脑电图(EEG)的脑-计算机接口系统，用于个体手指级别的机器人手控制。使用EEGNet深度学习架构进行运动想象(MI)和运动执行(ME)任务的多手指分类。

**论文**: Ding, Y., et al. (2025). Nature Communications. DOI: 10.1038/s41467-025-61064-x

## 常用命令

### PyTorch训练脚本 (推荐)

```bash
# 安装依赖
pip install -r requirements.txt

# 预训练模型 (PyTorch)
python train.py --subject 1 --session 1 --nclass 2 --task ME --mode pretrain

# 微调模型 (PyTorch)
python train.py --subject 1 --session 2 --nclass 2 --task ME --mode finetune

# 训练所有被试
python train.py --all-subjects --nclass 2 --task MI --mode pretrain

# 可视化结果
python visualize_results.py checkpoints/summary_ME_2class_pretrain.json
```

### 原始TensorFlow训练脚本 (兼容)

```bash
# 预训练模型
python main_model_training.py <subj_id> <session_num> <nclass> <task> <modeltype>

# 示例: 被试1，会话1，2类分类，运动执行任务，预训练
python main_model_training.py 1 1 2 ME Orig

# 示例: 被试1，会话2，3类分类，运动想象任务，微调
python main_model_training.py 1 2 3 MI Finetune
```

**参数说明**:
- `subj_id`: 被试ID (1-21)
- `session_num`: 会话号 (1-5)
- `nclass`: 分类数 (2或3)
- `task`: "MI"或"ME"
- `modeltype`: "Orig"(预训练) 或 "Finetune"(微调)

### 依赖安装 (推荐使用uv)

```bash
# 使用 uv (推荐 - 与EEG-BCI仓库相同)
uv venv
uv pip install -e ".[viz]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 或使用 pip
pip install -e ".[viz]"
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 原始TensorFlow依赖 (仅用于旧脚本)

```bash
pip install tensorflow>=2.0 numpy scipy scikit-learn mne pyriemann matplotlib
pip install BCPy2000==2021.1.0  # 仅在线处理需要
```

## 架构概览

```
# PyTorch实现 (推荐)
train.py                   # 统一训练入口 - PyTorch EEGNet实现
visualize_results.py       # 可视化脚本 - 生成图表和报告

# 原始TensorFlow实现
main_model_training.py     # 离线训练入口 - 配置参数并调用Functions
main_online_processing.py  # 在线处理入口 - BCPy2000信号处理模块
Functions.py               # 数据处理层 - 加载/滤波/分割/训练流程
EEGModels_tf.py           # 神经网络层 - EEGNet及变体的TensorFlow实现
```

### 数据处理流程

**离线训练**: MAT文件 → 事件提取 → 试验分割(1秒窗口) → CAR去噪 → 下采样(100Hz) → 带通滤波(4-40Hz) → Z-score标准化 → EEGNet

**在线处理**: 实时EEG流 → 去均值 → 重采样(100Hz) → 带通滤波 → Z-score → EEGNet推断 → 概率输出

### 关键配置

| 参数 | 预训练 | 微调 |
|------|--------|------|
| 学习率 | 0.001 | 1e-4 |
| Dropout | 0.5 | 0.65 |
| 训练轮数 | 300 | 100 |
| 微调层数 | - | 最后12层 |

### 手指标签映射

- 2类: 标签1(拇指) vs 标签4(小指)
- 3类: 标签1(拇指) vs 标签2(食指) vs 标签4(小指)

## 文件命名规范

PyTorch模型: `S{subj:02}_Sess{session:02}_{task}_{nclass}class_{mode}.pt`
示例: `S01_Sess01_ME_2class_pretrain.pt`

TensorFlow模型: `S{subj:02}_Sess{session:02}_{task}_{nclass}class_{modeltype}.h5`
示例: `S01_Sess01_ME_2class_Orig.h5`

## 技术栈

### PyTorch实现 (train.py)
- PyTorch 2.x
- SciPy (信号处理)
- scikit-learn (评估指标)
- matplotlib (可视化)

### 原始TensorFlow实现
- TensorFlow 2.x + Keras
- SciPy (信号处理)
- MNE (EEG数据处理)
- BCPy2000 (在线实时框架)

## 新特性 (PyTorch实现)

- 统一的命令行接口
- 自动设备检测 (CUDA/CPU)
- 彩色日志输出
- 多数投票准确率计算
- 训练曲线自动可视化
- JSON格式结果保存
- 批量训练所有被试

## 注意事项

- EEGNet实现来源于ARL EEGModels项目
- 数据格式为MAT文件，包含`eeg.data`和`eeg.fsample`字段
- 在线处理需要BCPy2000环境支持
- PyTorch版本需要数据放在`data/`目录下
