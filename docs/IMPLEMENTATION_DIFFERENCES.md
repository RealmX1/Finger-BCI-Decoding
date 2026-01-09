# 实现差异追踪文档
# Implementation Differences Tracker

> 本文档跟踪代码重新实现与原始论文之间的差异。
> This document tracks differences between our re-implementation and the original paper.

**状态图例 / Legend**:
- ✅ 完全对齐 (Fully Aligned)
- ⚠️ 存在差异 (Has Differences)
- ❌ 未实现 (Not Implemented)
- 🔄 待验证 (Needs Verification)

---

## 1. 模型架构 (Model Architecture)

### EEGNet-8,2

| 组件 | 论文 | 本实现 | 状态 | 备注 |
|------|------|--------|------|------|
| F1 (temporal filters) | 8 | 8 | ✅ | |
| D (depth multiplier) | 2 | 2 | ✅ | |
| F2 (pointwise filters) | 16 | 16 | ✅ | |
| kernLength | 32 | 32 | ✅ | |
| Dropout类型 | Dropout | Dropout | ✅ | |
| 激活函数 | ELU | ELU | ✅ | |
| Pooling | AvgPool (1,4), (1,8) | AvgPool (1,4), (1,8) | ✅ | |
| norm_rate | 0.25 | 0.25 | ✅ | Dense层max_norm |

**结论**: 模型架构 **完全对齐**

---

## 2. 数据预处理 (Data Preprocessing)

| 步骤 | 论文 | 本实现 | 状态 | 差异说明 |
|------|------|--------|------|----------|
| 重参考 | CAR | CAR | ✅ | `data - data.mean(axis=1)` |
| 下采样率 | 100 Hz | 100 Hz | ✅ | |
| 下采样方法 | 未明确指定 | scipy.signal.resample | 🔄 | 论文未明确说明方法 |
| 带通滤波范围 | 4-40 Hz | 4-40 Hz | ✅ | |
| 滤波器类型 | 4阶Butterworth | 4阶Butterworth | ✅ | |
| 滤波方法 | 因果滤波 | lfilter | ✅ | 适合实时处理 |
| 边界填充 | 零填充 | 100点零填充 | ✅ | |
| 标准化 | Z-score | Z-score (axis=2) | ✅ | |
| 窗口长度 | 1秒 | 1秒 | ✅ | |
| 滑动步长 | 125ms | 128 samples | ✅ | @1024Hz = 125ms |

**结论**: 数据预处理 **完全对齐**

---

## 3. 训练参数 (Training Parameters)

### 3.1 Base模型

| 参数 | 论文 | 本实现 | 状态 | 差异说明 |
|------|------|--------|------|----------|
| Epochs | 300 | 300 | ✅ | |
| Batch size | 16 | 16 | ✅ | |
| Optimizer | Adam | Adam (legacy) | ✅ | TF 2.10兼容性 |
| Learning rate | 0.001 | 0.001 | ✅ | |
| Dropout rate | 0.5 | 0.5 | ✅ | |
| Early stopping | patience=80 | patience=80 | ✅ | |
| LR reduction factor | 0.5 | 0.5 | ✅ | |
| LR reduction patience | 30 | 30 | ✅ | |
| Loss | categorical_crossentropy | categorical_crossentropy | ✅ | |
| 类别权重 | 动态balanced | compute_class_weight('balanced') | ✅ | |

### 3.2 Fine-tuned模型

| 参数 | 论文 | 本实现 | 状态 | 差异说明 |
|------|------|--------|------|----------|
| Epochs | 100 | 100 | ✅ | |
| Learning rate | 1e-4 | 1e-4 | ✅ | |
| Dropout rate | 更高 | 0.65 | ✅ | 论文未给具体值 |
| 冻结层数 | 前4层 | 前4层 | ✅ | |
| Early stopping | 未明确 | patience=30 | 🔄 | 论文未明确 |
| LR reduction patience | 未明确 | patience=15 | 🔄 | 论文未明确 |

**结论**: 训练参数 **基本对齐**，部分细节论文未明确说明

---

## 4. 在线平滑算法 (Online Smoothing)

| 组件 | 论文 (Eq. 1) | 本实现 | 状态 |
|------|--------------|--------|------|
| 初始状态 | h₀ = 0 | `self.h = np.zeros(n_classes)` | ✅ |
| 平滑公式 | P'ₜ = α×hₜ₋₁ + Pₜ | `p_prime = alpha * h + current_prob` | ✅ |
| 状态更新 | hₜ = P'ₜ | `self.h = p_prime.copy()` | ✅ |
| 归一化 | L2归一化 | `p_prime / np.linalg.norm(p_prime)` | ✅ |
| 概率归一化 | 和为1 | `p_prime / p_prime.sum()` | ✅ |
| 默认α值 | 0.5 | 0.5 | ✅ |

**结论**: 在线平滑算法 **完全对齐**

---

## 5. 评估方法 (Evaluation Methods)

| 指标 | 论文 | 本实现 | 状态 | 位置 |
|------|------|--------|------|------|
| Majority Voting | ✓ | ✓ | ✅ | `evaluation/test_evaluation.py:132-156` |
| Segment Accuracy | ✓ | ✓ | ✅ | `evaluation/test_evaluation.py:208` |
| Precision (每类) | ✓ | ✓ | ✅ | sklearn.metrics |
| Recall (每类) | ✓ | ✓ | ✅ | sklearn.metrics |
| 混淆矩阵 | ✓ | ✓ | ✅ | sklearn.metrics |

**结论**: 评估方法 **完全对齐**

---

## 6. 实验设计差异 (Experimental Design Differences)

### 6.1 已实现功能

| 功能 | 论文 | 本实现 | 状态 |
|------|------|--------|------|
| 离线训练 | ✓ | ✓ | ✅ |
| 单session训练 | ✓ | ✓ | ✅ |
| 多session累积训练 | ✓ | ✓ | ✅ |
| Fine-tuning | ✓ | ✓ | ✅ |
| 5折交叉验证 | ✓ (离线) | ✓ | ✅ |
| Majority Voting评估 | ✓ | ✓ | ✅ |

### 6.2 部分实现/需要外部环境

| 功能 | 论文 | 本实现 | 状态 | 说明 |
|------|------|--------|------|------|
| BCPy2000实时处理 | ✓ | 框架代码存在 | ⚠️ | 需要BCPy2000环境 |
| 机器人手控制 | ✓ | 未包含 | ❌ | 需要Allegro Hand硬件 |
| 视觉反馈界面 | ✓ | 未包含 | ❌ | 需要BCI2000 |

### 6.3 未实现功能

| 功能 | 论文 | 状态 | 优先级 |
|------|------|------|--------|
| ERD分析 | ✓ | ❌ | 低 |
| Saliency Map生成 | ✓ | ❌ | 低 |
| FBCSP-LDA基线 | ✓ | ❌ | 中 |
| deepEEGNet变体 | Supplementary | ❌ | 低 |

---

## 7. 数据格式差异 (Data Format Differences)

| 方面 | 论文 | 本实现 | 状态 |
|------|------|--------|------|
| EEG通道数 | 128 | 支持任意 | ✅ |
| 采样率 | 1024 Hz | 从数据自动读取 | ✅ |
| 数据格式 | MATLAB .mat | MATLAB .mat | ✅ |
| 事件标记 | Target/TrialEnd | Target/TrialEnd | ✅ |
| 手指标签 | 1=拇指,2=食指,3=中指,4=小指 | 相同 | ✅ |

---

## 8. 已知差异详情 (Known Differences Details)

### 8.1 TensorFlow版本兼容性

**差异**: 使用`tf.keras.optimizers.legacy.Adam`

**原因**: TensorFlow 2.10兼容性要求

**影响**: 无功能影响，仅API差异

**代码位置**: `Functions.py:348-354`, `training/cross_validation.py:216`

```python
# 本实现
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# TF 2.11+ 可改为
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

### 8.2 Fine-tuning Dropout率

**差异**: 论文描述为"更高"，本实现使用0.65

**原因**: 论文未给出具体数值

**影响**: 可能轻微影响性能

**验证状态**: 🔄 需要敏感性分析

### 8.3 Early Stopping Patience (Fine-tuning)

**差异**: Fine-tuning使用patience=30，论文未明确

**原因**: 根据epochs=100的合理推断

**影响**: 可能影响训练收敛

**验证状态**: 🔄 需要验证

---

## 9. 性能对比 (Performance Comparison)

### 9.1 论文报告性能

| 条件 | 任务 | 准确率 |
|------|------|--------|
| MI Session 2 Fine-tuned | 2-class | 80.56% |
| MI Session 2 Fine-tuned | 3-class | 60.61% |
| ME Session 2 Fine-tuned | 2-class | 81.10% |
| ME Session 2 Fine-tuned | 3-class | 60.11% |

### 9.2 本实现验证结果

| 条件 | 任务 | 准确率 | 状态 |
|------|------|--------|------|
| - | - | - | 🔄 待测试 |

> **注**: 需要使用公开数据集验证实现正确性

---

## 10. 待验证项目 (Items to Verify)

### 高优先级

- [ ] 使用公开数据验证离线解码性能
- [ ] 验证fine-tuning dropout率(0.65)的影响
- [ ] 对比5折CV结果与论文离线结果

### 中优先级

- [ ] 实现FBCSP-LDA基线用于对比
- [ ] 验证不同early stopping patience的影响

### 低优先级

- [ ] 实现ERD分析脚本
- [ ] 实现Saliency Map可视化
- [ ] 实现deepEEGNet变体

---

## 11. 变更日志 (Changelog)

### v1.0.0 (2025-01-09)
- 初始文档创建
- 完成论文方法与代码实现的全面对比
- 识别主要对齐项和潜在差异

---

## 12. 参考文件映射 (File Reference Mapping)

| 论文章节 | 对应代码文件 |
|---------|-------------|
| Methods - EEGNet | `EEGModels_tf.py` |
| Methods - Online decoding | `main_online_processing.py` |
| Methods - Preprocessing | `preprocessing/signal_processing.py`, `Functions.py` |
| Methods - Training | `training/cross_validation.py`, `Functions.py` |
| Methods - Smoothing (Eq.1) | `online/online_smoothing.py` |
| Methods - Evaluation | `evaluation/test_evaluation.py` |

---

*文档版本: 1.0*
*最后更新: 2025-01-09*
*维护者: [Project Team]*
