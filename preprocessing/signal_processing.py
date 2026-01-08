# preprocessing/signal_processing.py
#
# EEG信号预处理共享模块
# EEG Signal Preprocessing Shared Module
#
# 本模块提供项目中通用的预处理函数，统一了Functions.py、training/cross_validation.py
# 和evaluation/test_evaluation.py中的重复代码。
#
# 论文参数 (Paper parameters):
# - 下采样 (Downsample): 100 Hz
# - 带通滤波 (Bandpass): 4-40 Hz, 4阶Butterworth滤波器
# - 窗口长度 (Window): 1秒
# - 滤波器填充 (Filter padding): 100个采样点
#
# 参考文献 (Reference):
# Ding, Y., et al. (2025). Nature Communications. DOI: 10.1038/s41467-025-61064-x

import numpy as np
import scipy.signal
import scipy.stats
from scipy.signal import resample
from dataclasses import dataclass
from typing import Optional, Tuple, Union


# ============================================================================
# 默认参数 (Default Parameters)
# ============================================================================

# 论文中使用的默认参数
DEFAULT_DOWNSAMPLE_RATE = 100  # Hz, 下采样目标频率
DEFAULT_BANDPASS_LOW = 4       # Hz, 带通滤波低频截止
DEFAULT_BANDPASS_HIGH = 40     # Hz, 带通滤波高频截止
DEFAULT_FILTER_ORDER = 4       # Butterworth滤波器阶数
DEFAULT_PADDING_LENGTH = 100   # 滤波器填充采样点数
DEFAULT_WINDOW_LENGTH = 1.0    # 秒, 分段窗口长度


@dataclass
class PreprocessingParams:
    """
    预处理参数数据类

    用于统一管理预处理参数，可以从字典初始化或使用默认值。

    属性:
        srate: 原始采样率 (Hz)
        downsrate: 目标下采样率 (Hz)
        windowlen: 窗口长度 (秒)
        bandpass_filt: 带通滤波频率范围 [低频, 高频] (Hz)
        filter_order: Butterworth滤波器阶数
        padding_length: 滤波器填充采样点数
    """
    srate: float = 1000.0
    downsrate: float = DEFAULT_DOWNSAMPLE_RATE
    windowlen: float = DEFAULT_WINDOW_LENGTH
    bandpass_filt: Tuple[float, float] = (DEFAULT_BANDPASS_LOW, DEFAULT_BANDPASS_HIGH)
    filter_order: int = DEFAULT_FILTER_ORDER
    padding_length: int = DEFAULT_PADDING_LENGTH

    @classmethod
    def from_dict(cls, params: dict) -> 'PreprocessingParams':
        """
        从字典创建参数对象

        参数:
            params: 参数字典，可包含以下键:
                - srate: 原始采样率
                - downsrate: 目标下采样率
                - windowlen: 窗口长度
                - bandpass_filt: 带通滤波频率范围

        返回:
            PreprocessingParams 实例
        """
        return cls(
            srate=params.get('srate', 1000.0),
            downsrate=params.get('downsrate', DEFAULT_DOWNSAMPLE_RATE),
            windowlen=params.get('windowlen', DEFAULT_WINDOW_LENGTH),
            bandpass_filt=tuple(params.get('bandpass_filt', (DEFAULT_BANDPASS_LOW, DEFAULT_BANDPASS_HIGH))),
            filter_order=params.get('filter_order', DEFAULT_FILTER_ORDER),
            padding_length=params.get('padding_length', DEFAULT_PADDING_LENGTH)
        )


# ============================================================================
# 核心预处理函数 (Core Preprocessing Functions)
# ============================================================================

def downsample_data(data: np.ndarray,
                    original_rate: float,
                    target_rate: float,
                    window_length: float) -> np.ndarray:
    """
    对EEG数据进行下采样

    使用scipy.signal.resample实现时域下采样，保持信号的频率特性。

    参数:
        data: 输入数据，形状为 (nTrial, nChan, nSample) 或 (nChan, nSample)
        original_rate: 原始采样率 (Hz)
        target_rate: 目标采样率 (Hz)
        window_length: 窗口长度 (秒)

    返回:
        下采样后的数据，时间轴采样点数变为 window_length * target_rate

    示例:
        >>> data = np.random.randn(100, 32, 1000)  # 100试次, 32通道, 1000采样点
        >>> downsampled = downsample_data(data, 1000, 100, 1.0)
        >>> print(downsampled.shape)  # (100, 32, 100)
    """
    desired_length = int(window_length * target_rate)

    # 处理2D和3D数据
    if data.ndim == 2:
        # (nChan, nSample) -> 添加trial维度
        data = data[np.newaxis, :, :]
        result = resample(data, desired_length, t=None, axis=2, window=None, domain='time')
        return result[0]
    else:
        # (nTrial, nChan, nSample)
        return resample(data, desired_length, t=None, axis=2, window=None, domain='time')


def apply_bandpass_filter(data: np.ndarray,
                          fs: float,
                          low: float = DEFAULT_BANDPASS_LOW,
                          high: float = DEFAULT_BANDPASS_HIGH,
                          order: int = DEFAULT_FILTER_ORDER,
                          padding_length: int = DEFAULT_PADDING_LENGTH) -> np.ndarray:
    """
    应用带通滤波器（带边界填充）

    使用4阶Butterworth带通滤波器，通过零填充减少边缘效应。
    滤波采用因果滤波器(lfilter)以保持实时处理兼容性。

    参数:
        data: 输入数据，形状为 (nTrial, nChan, nSample) 或 (nChan, nSample)
        fs: 采样率 (Hz)
        low: 低频截止 (Hz)，默认4 Hz
        high: 高频截止 (Hz)，默认40 Hz
        order: 滤波器阶数，默认4
        padding_length: 两端填充的采样点数，默认100

    返回:
        滤波后的数据，形状与输入相同

    注意:
        - 填充使用零值(constant padding)
        - 滤波后移除填充部分，返回与输入相同长度的数据
        - 使用lfilter进行因果滤波，适合实时处理场景

    示例:
        >>> data = np.random.randn(100, 32, 100)  # 100试次, 32通道, 100采样点
        >>> filtered = apply_bandpass_filter(data, fs=100, low=4, high=40)
        >>> print(filtered.shape)  # (100, 32, 100)
    """
    # 处理2D数据
    squeeze_output = False
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        squeeze_output = True

    # 零填充以减少边缘效应
    # Pad with zeros to reduce edge effects
    padded = np.pad(
        data,
        ((0, 0), (0, 0), (padding_length, padding_length)),
        'constant',
        constant_values=0
    )

    # 设计Butterworth带通滤波器
    # Design Butterworth bandpass filter
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass', fs=fs)

    # 应用滤波器（沿时间轴）
    # Apply filter along time axis
    filtered = scipy.signal.lfilter(b, a, padded, axis=-1)

    # 移除填充部分
    # Remove padding
    filtered = filtered[:, :, padding_length:-padding_length]

    if squeeze_output:
        return filtered[0]
    return filtered


def apply_zscore(data: np.ndarray,
                 axis: int = 2,
                 handle_nan: bool = True) -> np.ndarray:
    """
    应用Z-score标准化（带NaN处理）

    对每个试次的每个通道独立进行Z-score标准化，
    使其均值为0，标准差为1。

    参数:
        data: 输入数据，形状为 (nTrial, nChan, nSample) 或 (nChan, nSample)
        axis: 进行标准化的轴，默认2（时间轴）
        handle_nan: 是否处理NaN值，默认True
            - True: 使用nan_policy='omit'忽略NaN计算，然后将NaN替换为0
            - False: 标准zscore，NaN保持为NaN

    返回:
        标准化后的数据，形状与输入相同

    注意:
        - 对于恒定信号（标准差为0），zscore会产生NaN
        - 当handle_nan=True时，这些NaN会被替换为0

    示例:
        >>> data = np.random.randn(100, 32, 100)
        >>> normalized = apply_zscore(data, axis=2)
        >>> print(np.mean(normalized, axis=2).mean())  # 接近0
    """
    # 处理2D数据的轴参数
    if data.ndim == 2 and axis == 2:
        axis = 1

    if handle_nan:
        # 使用nan_policy='omit'进行zscore计算
        normalized = scipy.stats.zscore(data, axis=axis, nan_policy='omit')
        # 将剩余的NaN替换为0
        normalized = np.nan_to_num(normalized, nan=0.0)
    else:
        normalized = scipy.stats.zscore(data, axis=axis)

    return normalized


def reshape_for_eegnet(data: np.ndarray) -> np.ndarray:
    """
    将数据reshape为EEGNet输入格式

    EEGNet期望的输入格式为 (nTrial, nChan, nSample, 1)，
    最后一个维度是kernels维度。

    参数:
        data: 输入数据，形状为 (nTrial, nChan, nSample)

    返回:
        重塑后的数据，形状为 (nTrial, nChan, nSample, 1)

    示例:
        >>> data = np.random.randn(100, 32, 100)
        >>> reshaped = reshape_for_eegnet(data)
        >>> print(reshaped.shape)  # (100, 32, 100, 1)
    """
    if data.ndim == 3:
        nTrial, nChan, nSample = data.shape
        return data.reshape(nTrial, nChan, nSample, 1)
    elif data.ndim == 4:
        # 已经是正确格式
        return data
    else:
        raise ValueError(f"输入数据维度错误: 期望3D或4D，得到{data.ndim}D")


# ============================================================================
# 主预处理流水线 (Main Preprocessing Pipeline)
# ============================================================================

def preprocess_eeg_data(data: np.ndarray,
                        params: Union[dict, PreprocessingParams],
                        reshape_output: bool = True,
                        verbose: bool = False) -> np.ndarray:
    """
    EEG数据主预处理流水线

    执行完整的预处理流程:
    1. 下采样 (Downsampling) - 使用scipy.signal.resample
    2. 带通滤波 (Bandpass filtering) - 4-40Hz, 4阶Butterworth
    3. Z-score标准化 (Z-score normalization)
    4. 重塑为EEGNet输入格式 (可选)

    参数:
        data: 输入数据，形状为 (nTrial, nChan, nSample)
            - nTrial: 试次数量
            - nChan: EEG通道数量
            - nSample: 时间采样点数量
        params: 预处理参数，可以是字典或PreprocessingParams对象
            必需的键:
            - srate: 原始采样率 (Hz)
            - downsrate: 目标下采样率 (Hz)，论文使用100 Hz
            - windowlen: 窗口长度 (秒)，论文使用1秒
            - bandpass_filt: 带通滤波频率范围 [low, high]，论文使用[4, 40] Hz
            可选的键:
            - filter_order: 滤波器阶数，默认4
            - padding_length: 滤波器填充长度，默认100
        reshape_output: 是否将输出reshape为EEGNet格式 (nTrial, nChan, nSample, 1)
            默认True
        verbose: 是否打印处理过程信息，默认False

    返回:
        预处理后的数据:
        - 如果reshape_output=True: 形状为 (nTrial, nChan, nSample_new, 1)
        - 如果reshape_output=False: 形状为 (nTrial, nChan, nSample_new)
        其中 nSample_new = windowlen * downsrate

    示例:
        >>> # 使用字典参数
        >>> params = {
        ...     'srate': 1000,
        ...     'downsrate': 100,
        ...     'windowlen': 1.0,
        ...     'bandpass_filt': [4, 40]
        ... }
        >>> data = np.random.randn(100, 32, 1000)  # 100试次, 32通道, 1秒@1000Hz
        >>> processed = preprocess_eeg_data(data, params)
        >>> print(processed.shape)  # (100, 32, 100, 1)

        >>> # 使用PreprocessingParams对象
        >>> pp = PreprocessingParams(srate=1000, downsrate=100)
        >>> processed = preprocess_eeg_data(data, pp)

    注意:
        - 此函数假设输入数据已经过分段(segmentation)
        - 分段操作(segment_data)应在调用此函数之前完成
        - 滤波使用因果滤波器(lfilter)，适合实时处理
    """
    # 将字典参数转换为PreprocessingParams
    if isinstance(params, dict):
        pp = PreprocessingParams.from_dict(params)
    else:
        pp = params

    if verbose:
        print(f"预处理参数:")
        print(f"  原始采样率: {pp.srate} Hz")
        print(f"  目标采样率: {pp.downsrate} Hz")
        print(f"  窗口长度: {pp.windowlen} 秒")
        print(f"  带通滤波: {pp.bandpass_filt[0]}-{pp.bandpass_filt[1]} Hz")
        print(f"  输入数据形状: {data.shape}")

    # 步骤1: 下采样
    # Step 1: Downsample
    data = downsample_data(
        data,
        original_rate=pp.srate,
        target_rate=pp.downsrate,
        window_length=pp.windowlen
    )

    if verbose:
        print(f"  下采样后: {data.shape}")

    # 步骤2: 带通滤波
    # Step 2: Bandpass filtering
    data = apply_bandpass_filter(
        data,
        fs=pp.downsrate,
        low=pp.bandpass_filt[0],
        high=pp.bandpass_filt[1],
        order=pp.filter_order,
        padding_length=pp.padding_length
    )

    if verbose:
        print(f"  滤波后: {data.shape}")

    # 步骤3: Z-score标准化
    # Step 3: Z-score normalization
    data = apply_zscore(data, axis=2, handle_nan=True)

    if verbose:
        print(f"  标准化后: {data.shape}")

    # 步骤4: 重塑为EEGNet输入格式 (可选)
    # Step 4: Reshape for EEGNet input (optional)
    if reshape_output:
        data = reshape_for_eegnet(data)
        if verbose:
            print(f"  重塑后: {data.shape}")

    return data


# ============================================================================
# 便捷函数 (Convenience Functions)
# ============================================================================

def preprocess_train_val_data(X_train: np.ndarray,
                              X_val: np.ndarray,
                              params: Union[dict, PreprocessingParams],
                              verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    同时预处理训练集和验证集数据

    使用相同的参数对训练集和验证集进行预处理，确保处理一致性。

    参数:
        X_train: 训练数据 (nTrial, nChan, nSample)
        X_val: 验证数据 (nTrial, nChan, nSample)
        params: 预处理参数
        verbose: 是否打印处理信息

    返回:
        (X_train_processed, X_val_processed): 预处理后的训练和验证数据

    示例:
        >>> params = {'srate': 1000, 'downsrate': 100, 'windowlen': 1.0, 'bandpass_filt': [4, 40]}
        >>> X_train = np.random.randn(80, 32, 1000)
        >>> X_val = np.random.randn(20, 32, 1000)
        >>> X_train_p, X_val_p = preprocess_train_val_data(X_train, X_val, params)
    """
    if verbose:
        print("预处理训练数据...")
    X_train_processed = preprocess_eeg_data(X_train, params, reshape_output=True, verbose=verbose)

    if verbose:
        print("\n预处理验证数据...")
    X_val_processed = preprocess_eeg_data(X_val, params, reshape_output=True, verbose=verbose)

    return X_train_processed, X_val_processed


def get_default_params(srate: float = 1000.0) -> dict:
    """
    获取论文中使用的默认预处理参数

    参数:
        srate: 原始数据的采样率 (Hz)

    返回:
        包含默认预处理参数的字典

    示例:
        >>> params = get_default_params(srate=1000)
        >>> print(params)
        {'srate': 1000, 'downsrate': 100, 'windowlen': 1.0, 'bandpass_filt': [4, 40], ...}
    """
    return {
        'srate': srate,
        'downsrate': DEFAULT_DOWNSAMPLE_RATE,
        'windowlen': DEFAULT_WINDOW_LENGTH,
        'bandpass_filt': [DEFAULT_BANDPASS_LOW, DEFAULT_BANDPASS_HIGH],
        'filter_order': DEFAULT_FILTER_ORDER,
        'padding_length': DEFAULT_PADDING_LENGTH
    }
