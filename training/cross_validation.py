# training/cross_validation.py
#
# 5折分层交叉验证模块，用于Base模型训练
# 按照论文方法实现分层交叉验证

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import resample
import scipy.signal
import scipy.stats

import os
import sys
import json
from datetime import datetime

# 添加父目录到路径以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EEGModels_tf import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K

from Functions import segment_data, DEFAULT_STEP_SIZE
from utils.training_callbacks import TableEpochLogger


class StratifiedKFoldTrainer:
    """
    5折分层交叉验证训练器

    用于Base模型训练，确保每折中各类别比例一致。
    """

    def __init__(self, n_splits=5, random_state=42):
        """
        初始化训练器

        参数:
            n_splits: 折数，默认5折
            random_state: 随机种子，确保可复现性
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def preprocess_fold_data(self, X_train, Y_train, X_val, Y_val, params):
        """
        对单折数据进行预处理

        步骤:
            1. 滑动窗口分段
            2. 下采样
            3. 带通滤波
            4. Z-score标准化
            5. 转换为EEGNet输入格式

        参数:
            X_train: 训练数据 (nTrial, nChan, nSample)
            Y_train: 训练标签
            X_val: 验证数据
            Y_val: 验证标签
            params: 参数字典

        返回:
            处理后的 X_train, Y_train, X_val, Y_val
        """
        DesiredLen = int(params['windowlen'] * params['downsrate'])
        segment_size = int(params['windowlen'] * params['srate'])
        step_size = params.get('step_size', DEFAULT_STEP_SIZE)

        # 滑动窗口分段
        X_train, Y_train, _ = segment_data(X_train, Y_train, segment_size, step_size)
        X_val, Y_val, _ = segment_data(X_val, Y_val, segment_size, step_size)

        # 下采样
        X_train = resample(X_train, DesiredLen, t=None, axis=2, window=None, domain='time')
        X_val = resample(X_val, DesiredLen, t=None, axis=2, window=None, domain='time')

        # 带通滤波 (4-40Hz)
        padding_length = 100
        padded_train = np.pad(X_train, ((0, 0), (0, 0), (padding_length, padding_length)),
                             'constant', constant_values=0)
        padded_val = np.pad(X_val, ((0, 0), (0, 0), (padding_length, padding_length)),
                           'constant', constant_values=0)

        b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass',
                                   fs=params['downsrate'])
        X_train = scipy.signal.lfilter(b, a, padded_train, axis=-1)
        X_val = scipy.signal.lfilter(b, a, padded_val, axis=-1)

        X_train = X_train[:, :, padding_length:-padding_length]
        X_val = X_val[:, :, padding_length:-padding_length]

        # Z-score标准化
        X_train = scipy.stats.zscore(X_train, axis=2, nan_policy='omit')
        X_val = scipy.stats.zscore(X_val, axis=2, nan_policy='omit')

        # 处理NaN值
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        # 转换为EEGNet格式 (nTrial, nChan, nSample, 1)
        nChan = X_train.shape[1]
        X_train = X_train.reshape(X_train.shape[0], nChan, DesiredLen, 1)
        X_val = X_val.reshape(X_val.shape[0], nChan, DesiredLen, 1)

        # 标签转为one-hot编码
        Y_train = np_utils.to_categorical(Y_train - 1)
        Y_val = np_utils.to_categorical(Y_val - 1)

        return X_train, Y_train, X_val, Y_val

    def build_model(self, nclass, nChan, nSample, dropout_rate=0.5):
        """
        构建EEGNet模型

        参数:
            nclass: 类别数
            nChan: 通道数
            nSample: 样本点数
            dropout_rate: Dropout比率

        返回:
            编译好的EEGNet模型
        """
        model = EEGNet(
            nb_classes=nclass,
            Chans=nChan,
            Samples=nSample,
            dropoutRate=dropout_rate,
            kernLength=32,
            F1=8,
            D=2,
            F2=16,
            dropoutType='Dropout'
        )
        return model

    def train_with_cv(self, data, label, params, save_prefix):
        """
        执行5折交叉验证训练

        参数:
            data: 原始数据 (nTrial, nChan, nSample)
            label: 标签
            params: 参数字典
            save_prefix: 模型保存路径前缀

        返回:
            results: 每折的训练结果
            best_model_path: 最佳模型路径
        """
        K.set_image_data_format('channels_last')

        nTrial = len(data)
        nChan = data.shape[1]
        DesiredLen = int(params['windowlen'] * params['downsrate'])

        results = []
        best_val_acc = 0
        best_model_path = None

        print(f"开始5折交叉验证训练...")
        print(f"总试次数: {nTrial}, 通道数: {nChan}")

        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(data, label)):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*50}")

            # 获取训练和验证数据
            X_train = data[train_idx]
            Y_train = label[train_idx]
            X_val = data[val_idx]
            Y_val = label[val_idx]

            print(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

            # 预处理
            X_train, Y_train, X_val, Y_val = self.preprocess_fold_data(
                X_train, Y_train, X_val, Y_val, params
            )

            print(f"预处理后 - 训练: {X_train.shape}, 验证: {X_val.shape}")

            # 计算动态类别权重
            train_labels = np.argmax(Y_train, axis=1)
            class_weights_array = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weights = {i: w for i, w in enumerate(class_weights_array)}
            print(f"类别权重: {class_weights}")

            # 构建模型
            model = self.build_model(
                nclass=params['nclass'],
                nChan=nChan,
                nSample=DesiredLen,
                dropout_rate=params.get('dropout_ratio', 0.5)
            )

            # 编译模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # 回调函数
            fold_save_path = f"{save_prefix}_fold{fold_idx + 1}.keras"
            # 回调函数 (使用表格日志)
            epoch_logger = TableEpochLogger(header_every=20, keep_every=5)
            callbacks = [
                ModelCheckpoint(
                    filepath=fold_save_path,
                    verbose=0,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True
                ),
                EarlyStopping(monitor='val_loss', patience=10),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                epoch_logger
            ]

            # 训练
            history = model.fit(
                X_train, Y_train,
                batch_size=16,
                epochs=params.get('epochs', 300),
                verbose=0,
                validation_data=(X_val, Y_val),
                callbacks=callbacks,
                class_weight=class_weights
            )

            # 记录结果
            fold_result = {
                'fold': fold_idx + 1,
                'best_val_accuracy': max(history.history['val_accuracy']),
                'best_val_loss': min(history.history['val_loss']),
                'model_path': fold_save_path,
                'history': {
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy'],
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss']
                }
            }
            results.append(fold_result)

            print(f"\nFold {fold_idx + 1} 最佳验证准确率: {fold_result['best_val_accuracy']:.4f}")

            # 更新最佳模型
            if fold_result['best_val_accuracy'] > best_val_acc:
                best_val_acc = fold_result['best_val_accuracy']
                best_model_path = fold_save_path

            # 清理内存
            K.clear_session()

        # 汇总结果
        mean_acc = np.mean([r['best_val_accuracy'] for r in results])
        std_acc = np.std([r['best_val_accuracy'] for r in results])

        print(f"\n{'='*50}")
        print(f"5折交叉验证完成!")
        print(f"平均验证准确率: {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(f"最佳模型: {best_model_path}")
        print(f"{'='*50}")

        # 保存交叉验证结果到JSON文件
        cv_results_path = f"{save_prefix}_cv_results.json"

        # 辅助函数：将NumPy类型转换为Python原生类型
        def convert_to_native(val):
            if val is None:
                return None
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.floating,)):
                return float(val)
            if isinstance(val, (list, tuple)):
                return [convert_to_native(v) for v in val]
            return val

        cv_results_data = {
            # 时间戳
            'timestamp': datetime.now().isoformat(),
            # 训练参数
            'training_params': {
                'n_splits': int(self.n_splits),
                'random_state': int(self.random_state) if self.random_state is not None else None,
                'nclass': convert_to_native(params.get('nclass')),
                'windowlen': convert_to_native(params.get('windowlen')),
                'srate': convert_to_native(params.get('srate')),
                'downsrate': convert_to_native(params.get('downsrate')),
                'bandpass_filt': convert_to_native(params.get('bandpass_filt')),
                'dropout_ratio': float(params.get('dropout_ratio', 0.5)),
                'epochs': int(params.get('epochs', 300)),
                'step_size': convert_to_native(params.get('step_size', DEFAULT_STEP_SIZE))
            },
            # 各折结果（不包含完整训练历史以减小文件大小）
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_val_accuracy': float(r['best_val_accuracy']),
                    'best_val_loss': float(r['best_val_loss']),
                    'model_path': r['model_path']
                }
                for r in results
            ],
            # 汇总统计
            'summary': {
                'mean_val_accuracy': float(mean_acc),
                'std_val_accuracy': float(std_acc),
                'best_model_path': best_model_path
            }
        }

        try:
            with open(cv_results_path, 'w', encoding='utf-8') as f:
                json.dump(cv_results_data, f, indent=2, ensure_ascii=False)
            print(f"交叉验证结果已保存至: {cv_results_path}")
        except Exception as e:
            print(f"警告: 保存交叉验证结果失败: {e}")

        return results, best_model_path


def train_with_simple_split(data, label, params, save_name, val_ratio=0.2):
    """
    简单训练/验证划分训练（用于Finetune模型）

    参数:
        data: 原始数据
        label: 标签
        params: 参数字典
        save_name: 模型保存路径
        val_ratio: 验证集比例

    返回:
        save_name: 保存的模型路径
    """
    K.set_image_data_format('channels_last')

    nTrial = len(data)
    nChan = data.shape[1]
    DesiredLen = int(params['windowlen'] * params['downsrate'])
    segment_size = int(params['windowlen'] * params['srate'])
    step_size = params.get('step_size', DEFAULT_STEP_SIZE)

    # 随机划分
    shuffled_idx = np.random.permutation(nTrial)
    train_size = int((1 - val_ratio) * nTrial)
    train_idx = shuffled_idx[:train_size]
    val_idx = shuffled_idx[train_size:]

    X_train = data[train_idx]
    Y_train = label[train_idx]
    X_val = data[val_idx]
    Y_val = label[val_idx]

    # 分段
    X_train, Y_train, _ = segment_data(X_train, Y_train, segment_size, step_size)
    X_val, Y_val, _ = segment_data(X_val, Y_val, segment_size, step_size)

    # 下采样
    X_train = resample(X_train, DesiredLen, t=None, axis=2, window=None, domain='time')
    X_val = resample(X_val, DesiredLen, t=None, axis=2, window=None, domain='time')

    # 带通滤波
    padding_length = 100
    padded_train = np.pad(X_train, ((0, 0), (0, 0), (padding_length, padding_length)),
                         'constant', constant_values=0)
    padded_val = np.pad(X_val, ((0, 0), (0, 0), (padding_length, padding_length)),
                       'constant', constant_values=0)

    b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass',
                               fs=params['downsrate'])
    X_train = scipy.signal.lfilter(b, a, padded_train, axis=-1)
    X_val = scipy.signal.lfilter(b, a, padded_val, axis=-1)

    X_train = X_train[:, :, padding_length:-padding_length]
    X_val = X_val[:, :, padding_length:-padding_length]

    # Z-score
    X_train = scipy.stats.zscore(X_train, axis=2, nan_policy='omit')
    X_val = scipy.stats.zscore(X_val, axis=2, nan_policy='omit')
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    # 转换格式
    X_train = X_train.reshape(X_train.shape[0], nChan, DesiredLen, 1)
    X_val = X_val.reshape(X_val.shape[0], nChan, DesiredLen, 1)

    Y_train = np_utils.to_categorical(Y_train - 1)
    Y_val = np_utils.to_categorical(Y_val - 1)

    # 动态类别权重
    train_labels = np.argmax(Y_train, axis=1)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}

    # 构建模型
    is_finetune = 'modelpath' in params
    dropout_rate = params.get('dropout_ratio', 0.65 if is_finetune else 0.5)

    model = EEGNet(
        nb_classes=params['nclass'],
        Chans=nChan,
        Samples=DesiredLen,
        dropoutRate=dropout_rate,
        kernLength=32,
        F1=8,
        D=2,
        F2=16,
        dropoutType='Dropout'
    )

    # 编译
    lr = 1e-4 if is_finetune else 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 微调：加载预训练权重并冻结部分层
    if is_finetune:
        print(f"加载预训练模型: {params['modelpath']}")
        model.load_weights(params['modelpath'])
        model.trainable = True

        # 冻结前4层
        layers_to_freeze = 4
        for layer in model.layers[:layers_to_freeze]:
            print(f"冻结层: {layer.name}")
            layer.trainable = False

    # 回调 (使用表格日志)
    epochs = params.get('epochs', 100 if is_finetune else 300)
    epoch_logger = TableEpochLogger(header_every=20, keep_every=5)
    callbacks = [
        ModelCheckpoint(filepath=save_name, verbose=0, monitor='val_accuracy',
                       mode='max', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=8 if is_finetune else 10),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4 if is_finetune else 5),
        epoch_logger
    ]

    # 训练
    model.fit(
        X_train, Y_train,
        batch_size=16,
        epochs=epochs,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        class_weight=class_weights
    )

    print(f"训练完成! 模型保存至: {save_name}")
    return save_name
