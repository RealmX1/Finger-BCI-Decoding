# utils/training_callbacks.py
"""
训练回调模块 (Training Callbacks Module)

提供自定义 Keras 回调，用于优化训练日志输出格式。
"""

import sys
import tensorflow as tf


class TableEpochLogger(tf.keras.callbacks.Callback):
    """
    表格化 Epoch 日志回调 (Tabular Epoch Logger Callback)

    特点:
    - 最新 epoch 始终可见
    - 每 5 个 epoch 保留输出，其他被下一个覆盖
    - 每 20 条保留日志后显示一次表头
    - 列对齐的表格格式

    使用示例:
        from utils.training_callbacks import TableEpochLogger

        model.fit(
            X_train, Y_train,
            callbacks=[TableEpochLogger()],
            verbose=0  # 禁用默认输出
        )
    """

    def __init__(self, header_every=20, keep_every=5):
        """
        参数:
            header_every: 每隔多少条保留日志显示表头 (默认20)
            keep_every: 每隔多少个epoch保留输出 (默认5)
        """
        super().__init__()
        self.header_every = header_every
        self.keep_every = keep_every
        self.kept_count = 0  # 已保留的日志计数

        # 列宽定义 (Column widths)
        self.col_widths = {
            'epoch': 7,
            'loss': 10,
            'acc': 10,
            'val_loss': 10,
            'val_acc': 10,
            'lr': 10
        }

    def _print_header(self):
        """打印表头"""
        header = (
            f"{'Epoch':>{self.col_widths['epoch']}} | "
            f"{'Loss':>{self.col_widths['loss']}} | "
            f"{'Acc':>{self.col_widths['acc']}} | "
            f"{'Val Loss':>{self.col_widths['val_loss']}} | "
            f"{'Val Acc':>{self.col_widths['val_acc']}} | "
            f"{'LR':>{self.col_widths['lr']}}"
        )
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)

    def _format_value(self, value, width):
        """格式化数值"""
        if value is None:
            return f"{'N/A':>{width}}"
        elif isinstance(value, float):
            if value < 0.0001:
                return f"{value:>{width}.2e}"
            else:
                return f"{value:>{width}.4f}"
        else:
            return f"{value:>{width}}"

    def _build_line(self, epoch, logs):
        """构建输出行"""
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        # 获取学习率
        lr = None
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        elif hasattr(self.model.optimizer, 'learning_rate'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        return (
            f"{epoch + 1:>{self.col_widths['epoch']}} | "
            f"{self._format_value(loss, self.col_widths['loss'])} | "
            f"{self._format_value(acc, self.col_widths['acc'])} | "
            f"{self._format_value(val_loss, self.col_widths['val_loss'])} | "
            f"{self._format_value(val_acc, self.col_widths['val_acc'])} | "
            f"{self._format_value(lr, self.col_widths['lr'])}"
        )

    def on_train_begin(self, logs=None):
        """训练开始时打印表头"""
        self.kept_count = 0
        self._print_header()

    def on_epoch_end(self, epoch, logs=None):
        """每个 epoch 结束时输出"""
        logs = logs or {}
        line = self._build_line(epoch, logs)

        is_keep_epoch = (epoch + 1) % self.keep_every == 0
        is_last_epoch = epoch + 1 == self.params.get('epochs', 0)

        if is_keep_epoch or is_last_epoch:
            # 清除当前行并打印保留行
            sys.stdout.write(f"\r{line}\n")
            sys.stdout.flush()
            self.kept_count += 1

            # 每 header_every 条保留日志后打印表头
            if self.kept_count % self.header_every == 0:
                self._print_header()
        else:
            # 覆盖显示当前进度
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()

    def on_train_end(self, logs=None):
        """训练结束"""
        pass  # 最后的换行已在 on_epoch_end 中处理


class ProgressCallback(tf.keras.callbacks.Callback):
    """
    简洁进度回调 (Simple Progress Callback)

    只显示关键信息：当前epoch、最佳验证准确率
    """

    def __init__(self):
        super().__init__()
        self.best_val_acc = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1

        # 简洁的进度显示
        sys.stdout.write(
            f"\rEpoch {epoch + 1:3d} | "
            f"val_acc: {val_acc:.4f} | "
            f"best: {self.best_val_acc:.4f} (ep {self.best_epoch})"
        )
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print(f"\n训练完成! 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
