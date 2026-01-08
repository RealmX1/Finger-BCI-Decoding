# utils/logging_config.py
"""
日志配置模块 (Logging Configuration Module)

为BCI项目提供标准化的日志记录功能，支持控制台和文件输出。
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_dir='logs', level=logging.INFO):
    """
    设置并返回一个配置好的日志记录器 (Setup and return a configured logger)

    参数 (Parameters):
        name: 日志记录器名称，通常使用 __name__
        log_dir: 日志文件保存目录，默认为 'logs'
        level: 日志级别，默认为 INFO
               可选: DEBUG, INFO, WARNING, ERROR, CRITICAL

    返回 (Returns):
        logger: 配置好的日志记录器实例

    使用示例 (Usage):
        from utils.logging_config import setup_logger
        logger = setup_logger(__name__)
        logger.info("训练开始")
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式: 时间戳 - 级别 - 模块名 - 消息
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器 (Console Handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器 (File Handler)
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件名包含日期
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """
    获取已存在的日志记录器或创建新的 (Get existing logger or create new one)

    这是一个便捷函数，使用默认配置获取日志记录器。

    参数 (Parameters):
        name: 日志记录器名称

    返回 (Returns):
        logger: 日志记录器实例
    """
    logger = logging.getLogger(name)

    # 如果记录器没有处理器，使用默认配置
    if not logger.handlers:
        return setup_logger(name)

    return logger
