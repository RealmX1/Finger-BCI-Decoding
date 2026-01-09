# WSL2 GPU 加速设置指南

由于 TensorFlow 在 Windows 原生环境下不再支持 CUDA（从 2.11 版本开始），推荐使用 WSL2 进行 GPU 加速训练。

> **注意**: 如果您使用 RTX 50 系列 (Blackwell) GPU，请参阅 [RTX50_GPU_SETUP.md](RTX50_GPU_SETUP.md)，其中包含针对 compute capability 12.0 的特殊配置。

## 适用范围

本指南适用于：
- RTX 40 系列 (Ada Lovelace)
- RTX 30 系列 (Ampere)
- RTX 20 系列 (Turing)

## 前置条件

- Windows 11 或 Windows 10 (21H2+)
- NVIDIA GPU
- 已安装 NVIDIA 驱动 (470.76+)

## 步骤 1: 安装 WSL2

```powershell
# 以管理员身份运行 PowerShell
wsl --install -d Ubuntu-22.04
```

重启后设置用户名和密码。

## 步骤 2: 验证 GPU 在 WSL2 中可见

```bash
# 在 WSL2 Ubuntu 中运行
nvidia-smi
```

应该能看到 RTX 5070 的信息。

## 步骤 3: 安装 uv 和 Python

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 验证
uv --version
```

## 步骤 4: 克隆项目并设置环境

```bash
# 进入 Windows 文件系统中的项目目录
cd /mnt/c/Users/zhang/Desktop/github/Finger-BCI-Decoding

# 创建 WSL2 专用的虚拟环境
uv venv .venv-wsl --python 3.10

# 激活环境
source .venv-wsl/bin/activate

# 安装带 GPU 支持的 TensorFlow
uv pip install tensorflow[and-cuda]==2.16.1

# 安装其他依赖
uv pip install numpy scipy scikit-learn matplotlib tqdm
```

## 步骤 5: 验证 GPU 支持

```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

应该输出类似:
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 步骤 6: 运行实验

```bash
cd /mnt/c/Users/zhang/Desktop/github/Finger-BCI-Decoding
python scripts/run_experiment.py --subj 1 --task MI --nclass 2 \
    --data-folder /mnt/c/Users/zhang/Desktop/github/EEG-BCI/data
```

## 常见问题

### Q: nvidia-smi 在 WSL2 中不工作
确保 Windows 驱动版本 >= 470.76，并且安装了 CUDA WSL 组件。

### Q: TensorFlow 找不到 GPU
尝试安装 CUDA toolkit:
```bash
# Ubuntu 22.04 + CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

### Q: 内存不足
在 `%USERPROFILE%\.wslconfig` 中配置:
```
[wsl2]
memory=16GB
swap=8GB
```

## 性能对比

| 配置 | 2000x2000 矩阵乘法 x10 |
|------|----------------------|
| Windows CPU (TF 2.10) | ~0.2秒 |
| WSL2 GPU (TF 2.16) | ~0.02秒 |

GPU 加速约 10 倍提升。

## 注意事项

1. 数据文件在 Windows 文件系统 (`/mnt/c/...`)，WSL2 可以直接访问
2. 模型训练建议使用 WSL2 GPU，开发调试可用 Windows CPU
3. 两个环境的虚拟环境是分开的 (`.venv` vs `.venv-wsl`)
