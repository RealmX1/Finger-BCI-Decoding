# 二元分类BCI实验复现实现计划

## 概述

本计划旨在完善 Finger-BCI-Decoding 仓库，使其能够完整复现论文《EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level》中的二元分类（拇指 vs 小指）实验。

**数据位置**: `C:\Users\zhang\Desktop\github\EEG-BCI\data`
**代码位置**: `C:\Users\zhang\Desktop\github\Finger-BCI-Decoding`

---

## 一、关键问题诊断

### 1.1 高优先级问题

| 问题 | 位置 | 影响 |
|------|------|------|
| 路径生成逻辑与实际数据目录不兼容 | `Functions.py:50-78` | **阻塞性** - 无法加载数据 |
| 缺失5折分层交叉验证 | `Functions.py:160-166` | **关键** - 不符合论文方法 |
| 缺失测试评估模块 | 整体缺失 | **关键** - 无法计算论文指标 |
| 缺失在线平滑机制 | `main_online_processing.py` | **中等** - 影响在线控制稳定性 |

### 1.2 实际数据目录结构

```
data/S01/
├── OfflineImagery/                          # A1+A2 (离线MI数据)
├── OnlineImagery_Sess01_2class_Base/        # B1 (S1 Base测试)
├── OnlineImagery_Sess01_2class_Finetune/    # B2 (S1 Finetune测试)
├── OnlineImagery_Sess02_2class_Base/        # C1 (S2 Base测试)
└── OnlineImagery_Sess02_2class_Finetune/    # C2 (S2 Finetune测试)
```

---

## 二、实现任务清单

### Phase 1: 数据加载修复 (高优先级)

#### 任务1.1: 修复 `generate_paths` 函数
**文件**: `Functions.py:50-78`

**修改内容**:
```python
def generate_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')
    task_suffix = 'Imagery' if task == 'MI' else 'Movement'
    data_paths = []

    if model_type == 'Orig':  # Base模型
        # 1. 离线数据
        offline_dir = os.path.join(subject_folder, f'Offline{task_suffix}')
        if os.path.exists(offline_dir):
            data_paths.append(offline_dir)

        # 2. 之前session的在线数据
        for prev_session in range(1, session_num):
            for data_type in ['Base', 'Finetune']:
                online_dir = os.path.join(
                    subject_folder,
                    f'Online{task_suffix}_Sess{prev_session:02}_{nclass}class_{data_type}'
                )
                if os.path.exists(online_dir):
                    data_paths.append(online_dir)

    elif model_type == 'Finetune':  # 微调模型
        finetune_dir = os.path.join(
            subject_folder,
            f'Online{task_suffix}_Sess{session_num:02}_{nclass}class_Base'
        )
        if os.path.exists(finetune_dir):
            data_paths.append(finetune_dir)

    return data_paths
```

#### 任务1.2: 添加测试数据路径生成函数
**文件**: `Functions.py` (新增函数)

```python
def generate_test_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    """生成测试数据路径"""
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')
    task_suffix = 'Imagery' if task == 'MI' else 'Movement'

    test_dir = os.path.join(
        subject_folder,
        f'Online{task_suffix}_Sess{session_num:02}_{nclass}class_{model_type}'
    )
    return test_dir if os.path.exists(test_dir) else None
```

---

### Phase 2: 训练模块完善 (高优先级)

#### 任务2.1: 实现5折分层交叉验证
**文件**: 新建 `training/cross_validation.py`

**核心实现**:
```python
from sklearn.model_selection import StratifiedKFold

class StratifiedKFoldTrainer:
    def __init__(self, n_splits=5, random_state=42):
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def train_with_cv(self, X, y, model_builder, params, save_prefix):
        results = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
            # 训练每一折并保存结果
            ...
        return results, best_model_path
```

#### 任务2.2: 动态计算类别权重
**文件**: `Functions.py:241`

**修改内容**:
```python
# 替换硬编码
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight('balanced', classes=np.unique(Y_train.argmax(axis=1)), y=Y_train.argmax(axis=1))
class_weights = {i: w for i, w in enumerate(class_weights_array)}
```

#### 任务2.3: 澄清微调冻结逻辑
**文件**: `Functions.py:245-253`

当前代码逻辑实际正确（冻结前4层），但变量名具有误导性：
```python
# 当前：layers_fine_tune = 12（表示"要微调的层数"）
# 建议改为：layers_to_freeze = 4（更清晰表示"要冻结的层数"）
```

---

### Phase 3: 测试评估模块 (高优先级)

#### 任务3.1: 创建评估模块
**文件**: 新建 `evaluation/test_evaluation.py`

**核心功能**:
- **Majority Voting准确率**: 每个trial通过多段预测的多数投票
- **Precision/Recall**: 每个类别（拇指/小指）的精确率和召回率
- **混淆矩阵**: 详细的分类结果

```python
class BinaryClassEvaluator:
    def evaluate_session(self, X_test, y_test):
        # 1. 对每个trial进行滑动窗口预测
        # 2. Majority voting确定最终预测
        # 3. 计算各项指标
        return {
            'majority_voting_accuracy': accuracy,
            'precision': {'thumb': p0, 'pinky': p1},
            'recall': {'thumb': r0, 'pinky': r1},
            'confusion_matrix': cm
        }
```

---

### Phase 4: 在线平滑机制 (中优先级)

#### 任务4.1: 实现平滑算法
**文件**: 新建 `online/online_smoothing.py`

**论文公式 Eq.(1)**:
```
h₀ = 0
P'ₜ = α * hₜ₋₁ + Pₜ
hₜ = P'ₜ
P'ₜ = P'ₜ / ||P'ₜ||
```

```python
class OnlineSmoother:
    def __init__(self, n_classes=2, alpha=0.5):
        self.alpha = alpha
        self.h = np.zeros(n_classes)

    def smooth(self, current_prob):
        p_prime = self.alpha * self.h + current_prob
        self.h = p_prime.copy()
        smoothed = p_prime / np.linalg.norm(p_prime)
        return smoothed / smoothed.sum()
```

#### 任务4.2: 集成到在线处理
**文件**: `main_online_processing.py`

在`Process`方法中调用平滑器。

---

### Phase 5: 主实验脚本 (高优先级)

#### 任务5.1: 创建完整实验脚本
**文件**: 新建 `scripts/run_experiment.py`

**功能**:
1. 训练Session1 Base (数据: A1+A2)
2. 测试Session1 Base (数据: B1)
3. 微调Session1 Finetune (数据: B1，基于S1 Base)
4. 测试Session1 Finetune (数据: B2)
5. 训练Session2 Base (数据: A1+A2+B1+B2)
6. 测试Session2 Base (数据: C1)
7. 微调Session2 Finetune (数据: C1，基于S2 Base)
8. 测试Session2 Finetune (数据: C2)
9. 保存所有结果为JSON

---

## 三、文件结构

```
Finger-BCI-Decoding/
├── Functions.py                    # [修改] 路径生成、类别权重
├── EEGModels_tf.py                # [保持] EEGNet模型
├── main_model_training.py         # [修改] 集成交叉验证
├── main_online_processing.py      # [修改] 集成平滑机制
├── training/
│   └── cross_validation.py        # [新建] 5折交叉验证
├── evaluation/
│   └── test_evaluation.py         # [新建] 测试评估模块
├── online/
│   └── online_smoothing.py        # [新建] 在线平滑
├── scripts/
│   └── run_experiment.py          # [新建] 主实验脚本
└── results/                       # [新建] 结果保存目录
```

---

## 四、实现优先级

| 优先级 | 任务 | 文件 | 预计工时 |
|--------|------|------|----------|
| **P0** | 修复路径生成 | Functions.py | 2h |
| **P0** | 5折交叉验证 | training/cross_validation.py | 3h |
| **P0** | 测试评估模块 | evaluation/test_evaluation.py | 4h |
| **P0** | 主实验脚本 | scripts/run_experiment.py | 4h |
| **P1** | 动态类别权重 | Functions.py | 1h |
| **P1** | 在线平滑机制 | online/online_smoothing.py | 3h |
| **P2** | 集成平滑到在线处理 | main_online_processing.py | 2h |

**总预计工时**: 约19小时

---

## 五、验证方法

### 5.1 单元测试
```bash
# 测试数据加载
python -c "from Functions import generate_paths; print(generate_paths(1, 'MI', 2, 1, 'Orig', 'path/to/data'))"

# 测试评估模块
python -c "from evaluation.test_evaluation import BinaryClassEvaluator; print('OK')"
```

### 5.2 集成测试
```bash
# 运行单被试实验
python scripts/run_experiment.py --subj 1 --task MI
```

### 5.3 结果验证标准

根据论文报告，二元分类MI任务预期准确率：
- Session 1 Base: ~70-75%
- Session 1 Finetune: ~75-80%
- Session 2 Finetune: **~80.56%** (论文报告值)

验证标准：
1. 微调后准确率应高于Base模型
2. Session 2应高于Session 1
3. 准确率应显著高于随机水平(50%)

---

## 六、关键参数参考

| 参数 | Base模型 | Finetune模型 |
|------|----------|--------------|
| Learning Rate | 0.001 | 0.0001 |
| Epochs | 300 | 100 |
| Dropout | 0.5 | 0.65 |
| Early Stopping | patience=80 | patience=30 |
| 冻结层 | 无 | 前4层 |
| 交叉验证 | 5折分层 | 80/20划分 |

---

## 七、待修改的关键文件

1. **`Functions.py`** - 修复路径生成、添加动态类别权重
2. **`main_model_training.py`** - 集成5折交叉验证（可选）
3. **`main_online_processing.py`** - 集成在线平滑
4. **新建文件**:
   - `training/cross_validation.py`
   - `evaluation/test_evaluation.py`
   - `online/online_smoothing.py`
   - `scripts/run_experiment.py`
