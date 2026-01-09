# RTX 50 系列 GPU 设置指南

本指南适用于 NVIDIA RTX 50 系列 (Blackwell 架构) GPU，包括 RTX 5070、5080、5090 等，compute capability 12.0+。

## 背景

RTX 50 系列使用了全新的 Blackwell 架构 (compute capability 12.0)，目前稳定版 TensorFlow 尚未包含预编译的 GPU kernel。解决方案是使用 TensorFlow Nightly 版本，其中包含了对新架构的支持。

## 环境要求

- Linux (Ubuntu 22.04/24.04 推荐) 或 WSL2
- NVIDIA 驱动 >= 550
- Conda 或 Miniconda

## 安装步骤

### 步骤 1: 创建 Conda 环境

```bash
conda create --name tf_gpu python=3.11.4 pip -y
conda activate tf_gpu
```

### 步骤 2: 安装 TensorFlow Nightly

```bash
pip install tf-nightly
```

Nightly 版本包含最新的 GPU kernel 支持，可以在 compute capability 12.0 上运行。

### 步骤 3: 安装 CUDA 库

通过 pip 安装 NVIDIA CUDA 库（无需系统级安装）：

```bash
pip install nvidia-cudnn-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cuda-nvrtc-cu12
```

### 步骤 4: 配置库路径

```bash
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

# 重新激活环境使配置生效
conda deactivate
conda activate tf_gpu
```

### 步骤 5: 安装项目依赖

```bash
pip install scipy scikit-learn matplotlib
```

### 步骤 6: 验证安装

```bash
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
"
```

预期输出：
```
TensorFlow version: 2.21.0-dev20260108
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 性能测试

```bash
python -c "
import tensorflow as tf
import time

with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    _ = tf.matmul(a, b)  # 预热

    start = time.time()
    for _ in range(100):
        c = tf.matmul(a, b)
    tf.test.experimental.sync_devices()
    print(f'GPU matmul (100x): {time.time()-start:.3f}s')

with tf.device('/CPU:0'):
    start = time.time()
    for _ in range(100):
        c = tf.matmul(a, b)
    print(f'CPU matmul (100x): {time.time()-start:.3f}s')
"
```

RTX 5070 典型结果：
- GPU: ~0.008s
- CPU: ~0.31s
- 加速比: **~39x**

## 运行训练

```bash
conda activate tf_gpu

# 完整实验
python scripts/run_experiment.py --subj 1 --task MI --nclass 2 --data-folder ./data

# 单独训练
python main_model_training.py 1 1 2 MI Orig --data-folder ./data --save-folder ./models
```

## 常见问题

### Q: 出现 JIT 编译警告

```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0a.
CUDA kernels will be jit-compiled from PTX...
```

**答**: 这是正常现象。虽然没有预编译 kernel，TensorFlow 会在运行时通过 XLA 编译。首次运行会稍慢，但后续运行会使用缓存。GPU 计算仍然正常工作且速度很快。

### Q: 找不到 libcudnn.so

确保已正确设置 `LD_LIBRARY_PATH`。运行：
```bash
echo $LD_LIBRARY_PATH
```
应包含 `nvidia/cudnn/lib` 路径。

### Q: GPU 未检测到

1. 检查 NVIDIA 驱动：`nvidia-smi`
2. 确保环境变量正确设置
3. 尝试重新激活 conda 环境

### Q: 内存不足 (OOM)

减小 batch size：
```python
batch_size = 8  # 默认 16
```

## 性能优化（可选）

### 混合精度训练

在代码开头添加：
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

可获得约 2x 额外加速。

### 增大 Batch Size

RTX 5070 有 12GB 显存，可以尝试：
```python
batch_size = 64  # 或 128
```

## 兼容性说明

| GPU 系列 | Compute Capability | 推荐方案 |
|---------|-------------------|---------|
| RTX 50 (Blackwell) | 12.0 | TensorFlow Nightly + 本指南 |
| RTX 40 (Ada) | 8.9 | TensorFlow 2.16+ |
| RTX 30 (Ampere) | 8.6 | TensorFlow 2.10+ |
| RTX 20 (Turing) | 7.5 | TensorFlow 2.10+ |

## 参考链接

- [TensorFlow GPU 支持](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA 兼容性](https://developer.nvidia.com/cuda-gpus)
- [Keras 3 文档](https://keras.io/keras_3/)
