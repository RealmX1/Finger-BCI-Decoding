# preprocessing/__init__.py
#
# 预处理模块 - 提供EEG信号预处理的共享函数
# Preprocessing module - shared functions for EEG signal preprocessing
#
# 根据论文参数:
# - 下采样: 100 Hz
# - 带通滤波: 4-40 Hz (4阶Butterworth)
# - 窗口长度: 1秒
# - 滤波器填充: 100个采样点

from .signal_processing import (
    preprocess_eeg_data,
    apply_bandpass_filter,
    apply_zscore,
    downsample_data,
    reshape_for_eegnet,
    preprocess_train_val_data,
    get_default_params,
    PreprocessingParams,
    # 默认参数常量
    DEFAULT_DOWNSAMPLE_RATE,
    DEFAULT_BANDPASS_LOW,
    DEFAULT_BANDPASS_HIGH,
    DEFAULT_FILTER_ORDER,
    DEFAULT_PADDING_LENGTH,
    DEFAULT_WINDOW_LENGTH
)

__all__ = [
    'preprocess_eeg_data',
    'apply_bandpass_filter',
    'apply_zscore',
    'downsample_data',
    'reshape_for_eegnet',
    'preprocess_train_val_data',
    'get_default_params',
    'PreprocessingParams',
    # 默认参数常量
    'DEFAULT_DOWNSAMPLE_RATE',
    'DEFAULT_BANDPASS_LOW',
    'DEFAULT_BANDPASS_HIGH',
    'DEFAULT_FILTER_ORDER',
    'DEFAULT_PADDING_LENGTH',
    'DEFAULT_WINDOW_LENGTH'
]
