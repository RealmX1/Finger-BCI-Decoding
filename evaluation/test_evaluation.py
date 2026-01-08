# evaluation/test_evaluation.py
#
# 测试评估模块，计算论文中的各项指标
# 包括：Majority Voting准确率、Precision、Recall、混淆矩阵

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.signal import resample
import scipy.signal
import scipy.stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from collections import Counter

import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Functions import segment_data, load_and_filter_data, DEFAULT_STEP_SIZE


class BinaryClassEvaluator:
    """
    二元分类评估器

    实现论文中的评估方法：
    1. 对每个trial进行滑动窗口分段
    2. 对每个分段进行预测
    3. 通过Majority Voting确定trial级别的最终预测
    4. 计算各项指标
    """

    def __init__(self, model_path, params, class_names=None):
        """
        初始化评估器

        参数:
            model_path: 训练好的模型路径
            params: 参数字典 (包含 srate, downsrate, windowlen, bandpass_filt, nclass)
            class_names: 类别名称列表，如 ['Thumb', 'Pinky']
        """
        self.model_path = model_path
        self.params = params
        self.model = None

        if class_names is None:
            if params['nclass'] == 2:
                self.class_names = ['Thumb', 'Pinky']
            else:
                self.class_names = ['Thumb', 'Index', 'Pinky']
        else:
            self.class_names = class_names

    def load_model(self):
        """加载训练好的模型"""
        from EEGModels_tf import EEGNet

        # 模型需要知道输入形状，但我们先用默认参数创建
        # 实际使用时会通过load_weights加载权重
        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False
        )
        print(f"模型加载成功: {self.model_path}")

    def preprocess_test_data(self, data, labels):
        """
        预处理测试数据

        参数:
            data: 原始测试数据 (nTrial, nChan, nSample)
            labels: 标签

        返回:
            X_test: 预处理后的数据
            Y_test: 分段后的标签
            trial_indices: 每个分段对应的原始trial索引
        """
        DesiredLen = int(self.params['windowlen'] * self.params['downsrate'])
        segment_size = int(self.params['windowlen'] * self.params['srate'])
        step_size = self.params.get('step_size', DEFAULT_STEP_SIZE)

        # 滑动窗口分段
        X_test, Y_test, trial_indices = segment_data(data, labels, segment_size, step_size)

        # 下采样
        X_test = resample(X_test, DesiredLen, t=None, axis=2, window=None, domain='time')

        # 带通滤波
        padding_length = 100
        padded = np.pad(X_test, ((0, 0), (0, 0), (padding_length, padding_length)),
                       'constant', constant_values=0)

        b, a = scipy.signal.butter(4, self.params['bandpass_filt'], btype='bandpass',
                                   fs=self.params['downsrate'])
        X_test = scipy.signal.lfilter(b, a, padded, axis=-1)
        X_test = X_test[:, :, padding_length:-padding_length]

        # Z-score标准化
        X_test = scipy.stats.zscore(X_test, axis=2, nan_policy='omit')
        X_test = np.nan_to_num(X_test, nan=0.0)

        # 转换为EEGNet格式
        nChan = X_test.shape[1]
        X_test = X_test.reshape(X_test.shape[0], nChan, DesiredLen, 1)

        return X_test, Y_test, trial_indices

    def predict_segments(self, X_test):
        """
        对分段数据进行预测

        参数:
            X_test: 预处理后的测试数据

        返回:
            predictions: 每个分段的预测类别
            probabilities: 每个分段的预测概率
        """
        if self.model is None:
            self.load_model()

        probabilities = self.model.predict(X_test, verbose=0)
        predictions = np.argmax(probabilities, axis=1) + 1  # 转换回1-indexed标签

        return predictions, probabilities

    def majority_voting(self, segment_predictions, trial_indices):
        """
        对每个trial应用Majority Voting

        参数:
            segment_predictions: 分段级别的预测
            trial_indices: 每个分段对应的trial索引

        返回:
            trial_predictions: trial级别的最终预测
        """
        unique_trials = np.unique(trial_indices)
        trial_predictions = []

        for trial_idx in unique_trials:
            # 获取该trial的所有分段预测
            mask = trial_indices == trial_idx
            preds = segment_predictions[mask]

            # Majority Voting
            counter = Counter(preds)
            majority_pred = counter.most_common(1)[0][0]
            trial_predictions.append(majority_pred)

        return np.array(trial_predictions)

    def evaluate_session(self, test_data_path, data_folder=None):
        """
        评估单个session的测试数据

        参数:
            test_data_path: 测试数据目录路径
            data_folder: 数据根目录（如果test_data_path是相对路径）

        返回:
            results: 包含各项指标的字典
        """
        K.set_image_data_format('channels_last')

        # 加载测试数据
        if data_folder:
            test_data_path = os.path.join(data_folder, test_data_path)

        print(f"加载测试数据: {test_data_path}")

        # 使用Functions.py中的数据加载函数
        test_data, test_labels, params_loaded = load_and_filter_data(
            [test_data_path], self.params
        )

        # 更新采样率（从加载的数据中获取）
        self.params['srate'] = params_loaded['srate']

        print(f"测试数据形状: {test_data.shape}")
        print(f"测试标签: {np.unique(test_labels, return_counts=True)}")

        # 保存trial级别的真实标签
        trial_labels = test_labels.copy()
        n_trials = len(trial_labels)

        # 预处理
        X_test, Y_test_seg, trial_indices = self.preprocess_test_data(test_data, test_labels)
        print(f"分段后样本数: {len(X_test)}")

        # 分段级别预测
        segment_predictions, segment_probs = self.predict_segments(X_test)

        # Trial级别预测（Majority Voting）
        trial_predictions = self.majority_voting(segment_predictions, trial_indices)

        # 确保trial数目一致
        assert len(trial_predictions) == n_trials, \
            f"Trial数目不匹配: 预测{len(trial_predictions)} vs 实际{n_trials}"

        # 计算指标
        # 分段级别
        segment_accuracy = accuracy_score(Y_test_seg, segment_predictions)

        # Trial级别（Majority Voting）
        majority_accuracy = accuracy_score(trial_labels, trial_predictions)

        # 每类Precision/Recall (trial级别)
        precision = precision_score(trial_labels, trial_predictions, average=None)
        recall = recall_score(trial_labels, trial_predictions, average=None)

        # 混淆矩阵
        cm = confusion_matrix(trial_labels, trial_predictions)

        # 详细分类报告
        report = classification_report(
            trial_labels, trial_predictions,
            target_names=self.class_names,
            output_dict=True
        )

        results = {
            'segment_accuracy': segment_accuracy,
            'majority_voting_accuracy': majority_accuracy,
            'precision': {self.class_names[i]: p for i, p in enumerate(precision)},
            'recall': {self.class_names[i]: r for i, r in enumerate(recall)},
            'confusion_matrix': cm.tolist(),
            'n_trials': n_trials,
            'n_segments': len(X_test),
            'classification_report': report,
            'trial_predictions': trial_predictions.tolist(),
            'trial_labels': trial_labels.tolist()
        }

        # 打印结果
        self._print_results(results)

        return results

    def _print_results(self, results):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"Trial数量: {results['n_trials']}")
        print(f"分段数量: {results['n_segments']}")
        print("-" * 60)
        print(f"分段级别准确率: {results['segment_accuracy']:.4f}")
        print(f"Majority Voting准确率: {results['majority_voting_accuracy']:.4f}")
        print("-" * 60)
        print("每类指标:")
        for class_name in self.class_names:
            print(f"  {class_name}:")
            print(f"    Precision: {results['precision'][class_name]:.4f}")
            print(f"    Recall: {results['recall'][class_name]:.4f}")
        print("-" * 60)
        print("混淆矩阵:")
        cm = np.array(results['confusion_matrix'])
        header = "     " + "  ".join([f"{name[:5]:>5}" for name in self.class_names])
        print(header)
        for i, row in enumerate(cm):
            row_str = f"{self.class_names[i][:5]:>5} " + "  ".join([f"{v:>5}" for v in row])
            print(row_str)
        print("=" * 60)


def evaluate_model(model_path, test_path, params, data_folder=None):
    """
    便捷函数：评估模型在测试集上的表现

    参数:
        model_path: 模型文件路径
        test_path: 测试数据目录
        params: 参数字典
        data_folder: 数据根目录

    返回:
        results: 评估结果字典
    """
    evaluator = BinaryClassEvaluator(model_path, params)
    results = evaluator.evaluate_session(test_path, data_folder)
    return results


def batch_evaluate(model_configs, data_folder, params):
    """
    批量评估多个模型

    参数:
        model_configs: 模型配置列表，每项包含:
            - model_path: 模型路径
            - test_path: 测试数据路径
            - name: 配置名称
        data_folder: 数据根目录
        params: 参数字典

    返回:
        all_results: 所有评估结果的字典
    """
    all_results = {}

    for config in model_configs:
        print(f"\n{'#' * 60}")
        print(f"评估: {config['name']}")
        print(f"{'#' * 60}")

        evaluator = BinaryClassEvaluator(config['model_path'], params)
        results = evaluator.evaluate_session(config['test_path'], data_folder)
        all_results[config['name']] = results

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    for name, results in all_results.items():
        print(f"{name}: {results['majority_voting_accuracy']:.4f}")

    return all_results
