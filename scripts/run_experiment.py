#!/usr/bin/env python
# scripts/run_experiment.py
#
# 主实验脚本：完整复现二元分类BCI实验
# 根据论文《EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control
# at Individual Finger Level》实现

import argparse
import json
import os
import sys
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Functions import generate_paths, generate_test_paths, load_and_filter_data
from training.cross_validation import StratifiedKFoldTrainer, train_with_simple_split
from evaluation.test_evaluation import BinaryClassEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行二元分类BCI实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    # 运行单个被试的完整实验
    python run_experiment.py --subj 1 --task MI --nclass 2

    # 指定数据和输出目录
    python run_experiment.py --subj 1 --task MI --data-folder /path/to/data --save-folder /path/to/save

    # 仅运行Session 1
    python run_experiment.py --subj 1 --task MI --sessions 1

    # 跳过交叉验证（使用简单划分）
    python run_experiment.py --subj 1 --task MI --no-cv
        '''
    )

    parser.add_argument('--subj', type=int, required=True,
                        help='被试ID (1-21)')
    parser.add_argument('--task', type=str, required=True, choices=['MI', 'ME'],
                        help='任务类型: MI (运动想象) 或 ME (运动执行)')
    parser.add_argument('--nclass', type=int, default=2, choices=[2, 3],
                        help='类别数 (默认: 2)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1, 2],
                        help='要运行的session列表 (默认: 1 2)')
    parser.add_argument('--data-folder', type=str,
                        default=r'C:\Users\zhang\Desktop\github\EEG-BCI\data',
                        help='数据根目录')
    parser.add_argument('--save-folder', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--no-cv', action='store_true',
                        help='跳过交叉验证，使用简单80/20划分')
    parser.add_argument('--epochs-base', type=int, default=300,
                        help='Base模型训练轮数 (默认: 300)')
    parser.add_argument('--epochs-finetune', type=int, default=100,
                        help='Finetune模型训练轮数 (默认: 100)')

    return parser.parse_args()


def get_default_params(nclass=2):
    """获取默认参数"""
    return {
        'maxtriallen': 5,       # 最大trial长度 (秒)
        'windowlen': 1,         # 滑动窗口长度 (秒)
        'block_size': 128,      # 块大小 (样本数)
        'downsrate': 100,       # 下采样率 (Hz)
        'bandpass_filt': [4, 40],  # 带通滤波 (Hz)
        'nclass': nclass,
        'dropout_ratio': 0.5,   # Base模型dropout
        'epochs': 300           # 默认训练轮数
    }


def train_base_model(subj_id, task, nclass, session_num, data_folder, save_folder,
                     params, use_cv=True):
    """
    训练Base模型

    参数:
        subj_id: 被试ID
        task: 任务类型
        nclass: 类别数
        session_num: session编号
        data_folder: 数据目录
        save_folder: 保存目录
        params: 参数字典
        use_cv: 是否使用交叉验证

    返回:
        model_path: 训练好的模型路径
        cv_results: 交叉验证结果 (若使用CV)
    """
    print(f"\n{'='*60}")
    print(f"训练 Base 模型 - S{subj_id:02} Session {session_num}")
    print(f"{'='*60}")

    # 生成训练数据路径
    data_paths = generate_paths(subj_id, task, nclass, session_num, 'Orig', data_folder)
    print(f"训练数据路径: {data_paths}")

    if not data_paths:
        print("警告: 未找到训练数据!")
        return None, None

    # 加载数据
    data, label, params = load_and_filter_data(data_paths, params)
    print(f"数据形状: {data.shape}, 标签: {np.unique(label, return_counts=True)}")

    # 模型保存路径
    model_prefix = os.path.join(
        save_folder,
        f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_Base'
    )

    if use_cv:
        # 5折交叉验证训练
        trainer = StratifiedKFoldTrainer(n_splits=5, random_state=42)
        cv_results, best_model_path = trainer.train_with_cv(
            data, label, params, model_prefix
        )
        return best_model_path, cv_results
    else:
        # 简单划分训练
        model_path = f"{model_prefix}.h5"
        params_copy = params.copy()
        params_copy['epochs'] = params.get('epochs_base', 300)
        train_with_simple_split(data, label, params_copy, model_path)
        return model_path, None


def train_finetune_model(subj_id, task, nclass, session_num, base_model_path,
                         data_folder, save_folder, params):
    """
    训练Finetune模型

    参数:
        subj_id: 被试ID
        task: 任务类型
        nclass: 类别数
        session_num: session编号
        base_model_path: 预训练Base模型路径
        data_folder: 数据目录
        save_folder: 保存目录
        params: 参数字典

    返回:
        model_path: 微调后的模型路径
    """
    print(f"\n{'='*60}")
    print(f"微调 Finetune 模型 - S{subj_id:02} Session {session_num}")
    print(f"{'='*60}")

    # 生成微调数据路径
    data_paths = generate_paths(subj_id, task, nclass, session_num, 'Finetune', data_folder)
    print(f"微调数据路径: {data_paths}")

    if not data_paths:
        print("警告: 未找到微调数据!")
        return None

    # 加载数据
    data, label, params = load_and_filter_data(data_paths, params)
    print(f"数据形状: {data.shape}, 标签: {np.unique(label, return_counts=True)}")

    # 模型保存路径
    model_path = os.path.join(
        save_folder,
        f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_Finetune.h5'
    )

    # 配置微调参数
    params_copy = params.copy()
    params_copy['modelpath'] = base_model_path
    params_copy['dropout_ratio'] = 0.65
    params_copy['epochs'] = params.get('epochs_finetune', 100)

    # 训练
    train_with_simple_split(data, label, params_copy, model_path)

    return model_path


def evaluate_model(model_path, subj_id, task, nclass, session_num, model_type,
                   data_folder, params):
    """
    评估模型

    参数:
        model_path: 模型路径
        subj_id: 被试ID
        task: 任务类型
        nclass: 类别数
        session_num: session编号
        model_type: 'Orig' (Base) 或 'Finetune'
        data_folder: 数据目录
        params: 参数字典

    返回:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估模型 - S{subj_id:02} Session {session_num} {model_type}")
    print(f"{'='*60}")

    # 获取测试数据路径
    test_path = generate_test_paths(subj_id, task, nclass, session_num, model_type, data_folder)

    if test_path is None:
        print("警告: 未找到测试数据!")
        return None

    print(f"测试数据路径: {test_path}")

    # 评估
    evaluator = BinaryClassEvaluator(model_path, params)
    results = evaluator.evaluate_session(test_path)

    return results


def run_single_session(subj_id, task, nclass, session_num, data_folder, save_folder,
                       params, use_cv=True, prev_base_path=None):
    """
    运行单个session的完整流程

    流程:
        1. 训练Base模型 (累积之前所有数据)
        2. 测试Base模型
        3. 微调Finetune模型
        4. 测试Finetune模型

    返回:
        session_results: session结果字典
        base_model_path: Base模型路径 (供下一session使用)
    """
    session_results = {
        'session': session_num,
        'base': {},
        'finetune': {}
    }

    # 1. 训练Base模型
    base_model_path, cv_results = train_base_model(
        subj_id, task, nclass, session_num, data_folder, save_folder,
        params, use_cv
    )
    session_results['base']['model_path'] = base_model_path
    session_results['base']['cv_results'] = cv_results

    # 2. 测试Base模型
    if base_model_path:
        base_test_results = evaluate_model(
            base_model_path, subj_id, task, nclass, session_num, 'Orig',
            data_folder, params
        )
        session_results['base']['test_results'] = base_test_results

    # 3. 微调Finetune模型
    if base_model_path:
        finetune_model_path = train_finetune_model(
            subj_id, task, nclass, session_num, base_model_path,
            data_folder, save_folder, params
        )
        session_results['finetune']['model_path'] = finetune_model_path

        # 4. 测试Finetune模型
        if finetune_model_path:
            finetune_test_results = evaluate_model(
                finetune_model_path, subj_id, task, nclass, session_num, 'Finetune',
                data_folder, params
            )
            session_results['finetune']['test_results'] = finetune_test_results

    return session_results, base_model_path


def run_experiment(args):
    """
    运行完整实验

    实验流程 (以2个session为例):
        Session 1:
            1. 训练S1 Base模型 (数据: 离线)
            2. 测试S1 Base (数据: B1)
            3. 微调S1 Finetune (数据: B1, 基于S1 Base)
            4. 测试S1 Finetune (数据: B2)

        Session 2:
            5. 训练S2 Base模型 (数据: 离线 + B1 + B2)
            6. 测试S2 Base (数据: C1)
            7. 微调S2 Finetune (数据: C1, 基于S2 Base)
            8. 测试S2 Finetune (数据: C2)
    """
    print("="*60)
    print("开始BCI二元分类实验")
    print("="*60)
    print(f"被试: S{args.subj:02}")
    print(f"任务: {args.task}")
    print(f"类别数: {args.nclass}")
    print(f"Sessions: {args.sessions}")
    print(f"数据目录: {args.data_folder}")
    print(f"保存目录: {args.save_folder}")
    print(f"使用交叉验证: {not args.no_cv}")
    print("="*60)

    # 创建保存目录
    os.makedirs(args.save_folder, exist_ok=True)

    # 初始化参数
    params = get_default_params(args.nclass)
    params['epochs_base'] = args.epochs_base
    params['epochs_finetune'] = args.epochs_finetune

    # 存储所有结果
    all_results = {
        'subject': args.subj,
        'task': args.task,
        'nclass': args.nclass,
        'timestamp': datetime.now().isoformat(),
        'sessions': {}
    }

    # 运行每个session
    prev_base_path = None
    for session_num in args.sessions:
        session_results, base_path = run_single_session(
            args.subj, args.task, args.nclass, session_num,
            args.data_folder, args.save_folder, params,
            use_cv=(not args.no_cv),
            prev_base_path=prev_base_path
        )
        all_results['sessions'][f'session_{session_num}'] = session_results
        prev_base_path = base_path

    # 保存结果
    result_filename = os.path.join(
        args.save_folder,
        f'results_S{args.subj:02}_{args.task}_{args.nclass}class.json'
    )
    with open(result_filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存至: {result_filename}")

    # 打印汇总
    print_summary(all_results)

    return all_results


def print_summary(results):
    """打印实验结果汇总"""
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)

    for session_key, session_data in results['sessions'].items():
        print(f"\n{session_key.upper()}:")
        print("-"*40)

        # Base结果
        if 'test_results' in session_data['base'] and session_data['base']['test_results']:
            base_acc = session_data['base']['test_results']['majority_voting_accuracy']
            print(f"  Base Majority Voting准确率: {base_acc:.4f}")

        # Finetune结果
        if 'test_results' in session_data['finetune'] and session_data['finetune']['test_results']:
            ft_acc = session_data['finetune']['test_results']['majority_voting_accuracy']
            print(f"  Finetune Majority Voting准确率: {ft_acc:.4f}")

    print("="*60)


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
