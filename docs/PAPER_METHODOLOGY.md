# 论文方法论文档
# Paper Methodology Documentation

> 本文档详细记录论文中描述的实验设计和方法论。
> This document details the experimental design and methodology described in the paper.

**论文**: Ding, Y., et al. (2025). EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. *Nature Communications*, 16, 5401.
**DOI**: [10.1038/s41467-025-61064-x](https://doi.org/10.1038/s41467-025-61064-x)

---

## 1. 实验概述 (Experimental Overview)

### 1.1 研究目标
- 实现基于EEG的非侵入式脑机接口，用于个体手指级别的机器人手实时控制
- 解码运动执行(ME)和运动想象(MI)任务中的单指运动

### 1.2 被试信息
- **总招募人数**: 49名右利手健康成年人
- **最终纳入人数**: 21名 (6男/15女，平均年龄: 24.23 ± 3.72岁)
- **排除标准**: 离线ME或MI二分类准确率 < 70%
- **前置经验**: 所有被试均有肢体级别MI-BCI经验（至少2小时训练）

### 1.3 实验阶段
```
Phase 1: ME Robotic Control (n=21)
  ├── 离线session × 1 (无反馈)
  └── 在线session × 2 (实时反馈)

Phase 2: MI Robotic Control (n=21)
  ├── 离线session × 1 (无反馈)
  └── 在线session × 2 (实时反馈)

Phase 3: MI Online Training (n=16)
  └── 额外在线session × 3

Phase 4: Online Smoothing (n=16)
  ├── ME session × 1
  └── MI session × 1
```

---

## 2. 任务范式 (Task Paradigms)

### 2.1 离线任务 (无反馈)

#### 运动执行 (ME)
- 重复单指屈伸运动（拇指、食指、中指、小指）
- 32 runs × 5 trials/finger/run = 640 trials
- Trial时长: 5秒 + 2秒间隔

#### 运动想象 (MI)
- 想象单指屈伸运动，不实际执行
- 相同的run和trial结构
- 指导：想象按钢琴键的感觉

### 2.2 在线任务 (实时反馈)

#### 二分类 (2-class)
- **手指**: 拇指 vs 小指
- 16 runs (前8: Base模型, 后8: Fine-tuned模型)
- 每run 10 trials/finger

#### 三分类 (3-class)
- **手指**: 拇指 vs 食指 vs 小指
- 16 runs (前8: Base模型, 后8: Fine-tuned模型)
- 每run 10 trials/finger

#### Trial时序
```
0s          1s                    3s       5s
|-----------|---------------------|--------|
  准备期        反馈期 (2s)           间隔期
            ↑
        反馈开始
        (视觉+机器人)
```

### 2.3 反馈机制
1. **视觉反馈**: 屏幕上目标手指变色
   - 绿色 = 分类正确
   - 红色 = 分类错误

2. **物理反馈**: Allegro机器人手
   - 实时移动解码出的手指
   - 每125ms更新一次

---

## 3. 数据采集 (Data Acquisition)

### 3.1 EEG设备
- **系统**: BioSemi ActiveTwo
- **通道数**: 128通道
- **采样率**: 1024 Hz
- **电极布置**: 国际10-20系统

### 3.2 实验环境
- 被试距屏幕约90cm
- 双手放在桌上的臂枕上
- 机器人手放置在被试和屏幕之间

---

## 4. 信号处理 (Signal Processing)

### 4.1 预处理流水线

```
原始EEG (1024 Hz, 128通道)
    ↓
[1] 共平均参考 (CAR)
    data = data - mean(data, axis=channels)
    ↓
[2] 滑动窗口分段
    窗口长度: 1秒
    步长: 125ms (128 samples)
    ↓
[3] 下采样
    目标频率: 100 Hz
    方法: scipy.signal.resample (时域)
    ↓
[4] 带通滤波
    频率范围: 4-40 Hz
    滤波器: 4阶Butterworth
    方法: lfilter (因果滤波)
    边界填充: 100采样点零填充
    ↓
[5] Z-score标准化
    按通道独立标准化
    ↓
[6] 重塑为EEGNet格式
    输出形状: (nTrial, nChan, nSample, 1)
```

### 4.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 下采样率 | 100 Hz | 目标频率 |
| 带通滤波 | 4-40 Hz | 包含alpha和beta频段 |
| 滤波器阶数 | 4 | Butterworth |
| 窗口长度 | 1秒 | 分段长度 |
| 步长 | 125ms | 滑动窗口步长 |
| 填充长度 | 100点 | 减少边缘效应 |

---

## 5. 深度学习模型 (Deep Learning Model)

### 5.1 EEGNet-8,2 架构

```
Input: (Chans, Samples, 1) = (128, 100, 1)
    ↓
[Block 1: 时域和空间滤波]
├── Conv2D(F1=8, kernel=(1, 32), padding='same')
├── BatchNormalization()
├── DepthwiseConv2D(D=2, kernel=(Chans, 1))
├── BatchNormalization()
├── Activation('elu')
├── AveragePooling2D((1, 4))
└── Dropout(0.5/0.65)
    ↓
[Block 2: 可分离卷积]
├── SeparableConv2D(F2=16, kernel=(1, 16), padding='same')
├── BatchNormalization()
├── Activation('elu')
├── AveragePooling2D((1, 8))
└── Dropout(0.5/0.65)
    ↓
[Classification Head]
├── Flatten()
├── Dense(nb_classes, kernel_constraint=max_norm(0.25))
└── Softmax()
```

### 5.2 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| F1 | 8 | 时域滤波器数量 |
| D | 2 | 深度乘数 (空间滤波器/时域滤波器) |
| F2 | 16 | 可分离卷积输出通道 (F1×D) |
| kernLength | 32 | 第一层卷积核长度 |
| norm_rate | 0.25 | Dense层的max_norm约束 |

---

## 6. 训练策略 (Training Strategy)

### 6.1 Base模型训练

| 参数 | 值 |
|------|-----|
| Epochs | 300 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Dropout rate | 0.5 |
| Early stopping | patience=80 |
| LR reduction | factor=0.5, patience=30 |
| Loss | categorical_crossentropy |
| 类别权重 | 动态计算 (balanced) |

**训练数据来源**:
- 离线session数据
- 之前所有在线session的数据 (累积)

### 6.2 Fine-tuned模型训练

| 参数 | 值 |
|------|-----|
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | **1e-4** (更小) |
| Dropout rate | **0.65** (更高) |
| Early stopping | patience=30 |
| LR reduction | factor=0.5, patience=15 |
| 冻结层数 | 前4层 |

**Fine-tuning数据**:
- 同一天session的前8 runs数据

### 6.3 层冻结策略

```python
# 冻结前4层 (temporal conv + spatial depthwise conv)
for layer in model.layers[:4]:
    layer.trainable = False
```

**冻结的层**:
1. Conv2D (时域卷积)
2. BatchNormalization
3. DepthwiseConv2D (空间滤波)
4. BatchNormalization

---

## 7. 在线解码 (Online Decoding)

### 7.1 实时处理流程

```
实时EEG流 (1024 Hz)
    ↓
每125ms:
    ├── 获取最近1秒数据
    ├── CAR重参考
    ├── 下采样到100Hz
    ├── 带通滤波 (4-40Hz)
    ├── Z-score标准化
    ├── EEGNet预测
    └── 输出概率分布
    ↓
机器人控制:
    概率最高的手指执行屈曲动作
```

### 7.2 在线平滑算法 (Equation 1)

```
h₀ = 0                          # 初始历史状态
P'ₜ = α × hₜ₋₁ + Pₜ            # 加权历史和当前概率
hₜ = P'ₜ                        # 更新历史状态
P'ₜ = P'ₜ / ||P'ₜ||            # L2归一化
```

**参数**:
- α = 0.5 (平滑系数)
- Pₜ: 当前原始概率分布
- P'ₜ: 平滑后概率分布

---

## 8. 评估指标 (Evaluation Metrics)

### 8.1 主要指标

| 指标 | 定义 |
|------|------|
| **Majority Voting Accuracy** | Trial级别准确率，基于分段预测的多数投票 |
| Segment Accuracy | 分段级别准确率 |
| Precision | 每类的精确率 |
| Recall | 每类的召回率 |

### 8.2 Majority Voting流程

```
单个Trial (3秒, 反馈期2秒)
    ↓
分段预测 (每125ms一次, 约16个预测)
    ↓
多数投票:
    final_prediction = mode(all_segment_predictions)
    ↓
与真实标签比较
```

---

## 9. 报告的性能 (Reported Performance)

### 9.1 MI任务 (n=21)

| 任务 | Session | 模型 | 准确率 |
|------|---------|------|--------|
| 2-class | 1 | Base | ~68% |
| 2-class | 1 | Fine-tuned | ~75% |
| 2-class | 2 | Fine-tuned | **80.56%** |
| 3-class | 1 | Base | ~47% |
| 3-class | 1 | Fine-tuned | ~55% |
| 3-class | 2 | Fine-tuned | **60.61%** |

### 9.2 ME任务 (n=21)

| 任务 | Session | 模型 | 准确率 |
|------|---------|------|--------|
| 2-class | 2 | Fine-tuned | **81.10%** |
| 3-class | 2 | Fine-tuned | **60.11%** |

### 9.3 统计显著性

- Session效应: p < 0.01 (2-way repeated measures ANOVA)
- 模型效应 (Base vs Fine-tuned): p < 0.001
- Fine-tuning在所有条件下显著提升性能

---

## 10. 离线解码性能 (Offline Decoding)

### 10.1 手指对组合 (EEGNet)

| 手指对 | MI准确率 | ME准确率 |
|--------|---------|---------|
| 拇指-小指 (1-4) | **77.58%** | **75.65%** |
| 拇指-食指 (1-2) | ~65% | ~63% |
| 食指-中指 (2-3) | ~55% | ~53% |
| 拇指-中指 (1-3) | ~68% | ~66% |

### 10.2 多分类

| 分类数 | MI准确率 | ME准确率 | 随机水平 |
|--------|---------|---------|---------|
| 4-class | 43.61% | 42.17% | 25% |

---

## 11. 电生理分析 (Electrophysiological Analysis)

### 11.1 ERD分析

**频段**:
- Alpha: 8-13 Hz
- Beta: 13-30 Hz

**主要发现**:
- MI/ME任务中左侧感觉运动区显著alpha/beta ERD
- 小指运动诱发最强ERD

### 11.2 Saliency Map分析

- 模型主要关注对侧hand knob区域
- 顶叶和枕叶区域也有贡献 (视觉注意)

---

## 12. 参考文献

1. Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.* 15, 056013.

2. Ang, K. K. et al. (2012). Filter bank common spatial pattern algorithm on BCI competition IV datasets 2a and 2b. *Front. Neurosci.* 6, 39.

---

*文档版本: 1.0*
*最后更新: 2025-01-09*
