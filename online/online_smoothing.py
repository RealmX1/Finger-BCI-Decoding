# online/online_smoothing.py
#
# 在线平滑机制实现
# 根据论文Equation (1)实现概率平滑

import numpy as np


class OnlineSmoother:
    """
    在线概率平滑器

    实现论文中的平滑算法 (Equation 1):
        h_0 = 0
        P'_t = alpha * h_{t-1} + P_t
        h_t = P'_t
        P'_t = P'_t / ||P'_t||

    其中:
        - h: 历史累积状态
        - P_t: 当前时刻的原始概率分布
        - P'_t: 平滑后的概率分布
        - alpha: 平滑系数 (0-1)

    平滑效果:
        - alpha = 0: 无平滑，直接输出当前概率
        - alpha = 1: 完全平滑，输出历史累积
        - 论文建议值: alpha = 0.5
    """

    def __init__(self, n_classes=2, alpha=0.5):
        """
        初始化平滑器

        参数:
            n_classes: 类别数量
            alpha: 平滑系数 (0-1)，越大平滑效果越强
        """
        self.n_classes = n_classes
        self.alpha = alpha
        self.h = np.zeros(n_classes)  # 历史状态
        self._initialized = False

    def reset(self):
        """重置平滑器状态（在新trial开始时调用）"""
        self.h = np.zeros(self.n_classes)
        self._initialized = False

    def smooth(self, current_prob):
        """
        对当前概率进行平滑

        参数:
            current_prob: 当前时刻的概率分布 (n_classes,)

        返回:
            smoothed_prob: 平滑后的概率分布
        """
        current_prob = np.asarray(current_prob).flatten()

        if len(current_prob) != self.n_classes:
            raise ValueError(f"概率维度不匹配: 期望{self.n_classes}, 实际{len(current_prob)}")

        # 应用平滑公式
        # P'_t = alpha * h_{t-1} + P_t
        p_prime = self.alpha * self.h + current_prob

        # 更新历史状态
        # h_t = P'_t
        self.h = p_prime.copy()

        # L2归一化
        # P'_t = P'_t / ||P'_t||
        norm = np.linalg.norm(p_prime)
        if norm > 0:
            p_prime = p_prime / norm

        # 转换为概率分布（确保和为1）
        smoothed_prob = p_prime / p_prime.sum()

        self._initialized = True
        return smoothed_prob

    def get_prediction(self, current_prob, threshold=None):
        """
        获取平滑后的预测结果

        参数:
            current_prob: 当前概率分布
            threshold: 可选的决策阈值，若最大概率低于此值则返回-1（不确定）

        返回:
            prediction: 预测类别 (0-indexed)
            smoothed_prob: 平滑后的概率分布
        """
        smoothed_prob = self.smooth(current_prob)

        max_prob = np.max(smoothed_prob)
        prediction = np.argmax(smoothed_prob)

        if threshold is not None and max_prob < threshold:
            prediction = -1  # 不确定

        return prediction, smoothed_prob


class AdaptiveSmoother(OnlineSmoother):
    """
    自适应平滑器

    基于预测置信度动态调整平滑系数
    高置信度时减少平滑，低置信度时增加平滑
    """

    def __init__(self, n_classes=2, alpha_base=0.5, alpha_range=(0.2, 0.8)):
        """
        初始化自适应平滑器

        参数:
            n_classes: 类别数量
            alpha_base: 基础平滑系数
            alpha_range: 平滑系数范围 (min, max)
        """
        super().__init__(n_classes, alpha_base)
        self.alpha_base = alpha_base
        self.alpha_min, self.alpha_max = alpha_range

    def smooth(self, current_prob):
        """
        自适应平滑

        置信度高时（概率分布偏向某一类），减小alpha（减少平滑）
        置信度低时（概率分布均匀），增大alpha（增加平滑）
        """
        current_prob = np.asarray(current_prob).flatten()

        # 计算置信度（使用熵的倒数作为指标）
        eps = 1e-10
        entropy = -np.sum(current_prob * np.log(current_prob + eps))
        max_entropy = np.log(self.n_classes)
        confidence = 1 - (entropy / max_entropy)  # 0-1，越高越确定

        # 自适应调整alpha
        # 高置信度时alpha变小，低置信度时alpha变大
        self.alpha = self.alpha_max - confidence * (self.alpha_max - self.alpha_min)

        # 调用父类的平滑方法
        return super().smooth(current_prob)


class TrialSmoother:
    """
    Trial级别平滑管理器

    管理单个trial内的连续预测平滑
    每个新trial开始时自动重置
    """

    def __init__(self, n_classes=2, alpha=0.5, smoother_type='standard'):
        """
        初始化Trial平滑管理器

        参数:
            n_classes: 类别数量
            alpha: 平滑系数
            smoother_type: 'standard' 或 'adaptive'
        """
        self.n_classes = n_classes
        self.alpha = alpha

        if smoother_type == 'adaptive':
            self.smoother = AdaptiveSmoother(n_classes, alpha)
        else:
            self.smoother = OnlineSmoother(n_classes, alpha)

        self.predictions = []
        self.probabilities = []

    def start_new_trial(self):
        """开始新的trial，重置平滑器"""
        self.smoother.reset()
        self.predictions = []
        self.probabilities = []

    def process_segment(self, segment_prob):
        """
        处理单个分段的预测概率

        参数:
            segment_prob: 分段的原始预测概率

        返回:
            prediction: 当前预测
            smoothed_prob: 平滑后的概率
        """
        prediction, smoothed_prob = self.smoother.get_prediction(segment_prob)

        self.predictions.append(prediction)
        self.probabilities.append(smoothed_prob.copy())

        return prediction, smoothed_prob

    def get_trial_prediction(self):
        """
        获取整个trial的最终预测（基于平滑后的概率）

        返回:
            final_prediction: 最终预测类别
            mean_prob: 平均概率分布
        """
        if not self.probabilities:
            return None, None

        mean_prob = np.mean(self.probabilities, axis=0)
        final_prediction = np.argmax(mean_prob)

        return final_prediction, mean_prob


# 便捷函数

def smooth_predictions(raw_probabilities, alpha=0.5, reset_indices=None):
    """
    对一序列原始概率进行平滑

    参数:
        raw_probabilities: 原始概率数组 (n_samples, n_classes)
        alpha: 平滑系数
        reset_indices: 需要重置平滑器的索引（如trial边界）

    返回:
        smoothed_probabilities: 平滑后的概率数组
    """
    n_samples, n_classes = raw_probabilities.shape
    smoother = OnlineSmoother(n_classes, alpha)
    smoothed = np.zeros_like(raw_probabilities)

    if reset_indices is None:
        reset_indices = set()
    else:
        reset_indices = set(reset_indices)

    for i in range(n_samples):
        if i in reset_indices:
            smoother.reset()
        smoothed[i] = smoother.smooth(raw_probabilities[i])

    return smoothed


if __name__ == "__main__":
    # 简单测试
    print("测试在线平滑器...")

    # 模拟预测概率序列
    np.random.seed(42)
    n_samples = 10
    raw_probs = np.random.dirichlet([1, 1], n_samples)  # 生成随机概率分布

    print("\n原始概率:")
    print(raw_probs)

    # 标准平滑
    smoother = OnlineSmoother(n_classes=2, alpha=0.5)
    smoothed = []
    for prob in raw_probs:
        smoothed.append(smoother.smooth(prob))
    smoothed = np.array(smoothed)

    print("\n平滑后概率 (alpha=0.5):")
    print(smoothed)

    # 对比预测
    raw_preds = np.argmax(raw_probs, axis=1)
    smoothed_preds = np.argmax(smoothed, axis=1)

    print("\n原始预测:", raw_preds)
    print("平滑预测:", smoothed_preds)

    print("\n测试完成!")
