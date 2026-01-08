#   main_model_training.py

#   This is the main deep learning training script used in the manuscript:
#   "EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level"
#
#   Takes in 5 required arguments and 2 optional arguments:
#   subj_id: (int) the subject ID, 1-21
#   session_num: (int) the session number, 1-5
#   nclass: (int) number of classes, 2 or 3
#   task: (string) motor imagery or execution, "ME" or "MI"
#   modeltype: (string) pre-training or fine-tuning, "Orig" or "Finetune"
#   --data-folder: (string, optional) path to data directory
#   --save-folder: (string, optional) path to save models directory
#
#   Example use:
#     python main_model_training.py 1 1 2 ME Orig
#     python main_model_training.py 1 1 2 ME Orig --data-folder /path/to/data --save-folder /path/to/save
#   Copyright (C) Yidan Ding 2025

# %%

from Functions import load_and_filter_data, generate_paths, train_models

import argparse
import os
import sys
import numpy as np


def parse_args():
    """解析命令行参数 (Parse command-line arguments)"""
    parser = argparse.ArgumentParser(
        description='训练EEG-BCI深度学习模型 (Train EEG-BCI deep learning model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例 (Examples):
    # 基础使用 (Basic usage)
    python main_model_training.py 1 1 2 ME Orig

    # 指定数据和保存目录 (Specify data and save directories)
    python main_model_training.py 1 1 2 ME Orig --data-folder /path/to/data --save-folder /path/to/save

    # 微调模型 (Fine-tuning)
    python main_model_training.py 1 1 2 ME Finetune --data-folder /path/to/data --save-folder /path/to/save
        '''
    )

    # 位置参数 (Positional arguments) - 保持向后兼容
    parser.add_argument('subj_id', type=int,
                        help='被试ID (Subject ID), 范围1-21')
    parser.add_argument('session_num', type=int,
                        help='Session编号 (Session number), 范围1-5')
    parser.add_argument('nclass', type=int, choices=[2, 3],
                        help='类别数 (Number of classes): 2 或 3')
    parser.add_argument('task', type=str, choices=['MI', 'ME'],
                        help='任务类型 (Task type): MI (运动想象/motor imagery) 或 ME (运动执行/motor execution)')
    parser.add_argument('modeltype', type=str, choices=['Orig', 'Finetune'],
                        help='模型类型 (Model type): Orig (预训练/pre-training) 或 Finetune (微调/fine-tuning)')

    # 可选参数 (Optional arguments)
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='数据根目录 (Data root directory), 默认: ./data')
    parser.add_argument('--save-folder', type=str, default='./models',
                        help='模型保存目录 (Model save directory), 默认: ./models')

    args = parser.parse_args()

    # 额外验证 (Additional validation)
    if args.subj_id < 1 or args.subj_id > 21:
        parser.error(f"subj_id必须在1-21范围内，当前值: {args.subj_id}")

    if args.session_num < 1 or args.session_num > 5:
        parser.error(f"session_num必须在1-5范围内，当前值: {args.session_num}")

    return args


# 解析命令行参数 (Parse command-line arguments)
args = parse_args()
subj_id = args.subj_id
session_num = args.session_num
nclass = args.nclass
task = args.task
modeltype = args.modeltype
data_folder = args.data_folder
save_folder = args.save_folder

# parameters
params = {
    'maxtriallen':5, # in s
    'windowlen':1, # in s
    'block_size': 128, # (samples) same as the online config
    'downsrate': 100, # dawnsampling rate
    'bandpass_filt': [4,40], # (Hz) bandpass filtering
    'nclass': nclass
}

# 创建保存目录 (Create save directory if not exists)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

data_paths = generate_paths(subj_id, task, nclass, session_num, model_type = modeltype, data_folder = data_folder)

data, label, params = load_and_filter_data(data_paths, params)

save_name = os.path.join(save_folder, f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5')

if modeltype == 'Finetune':
    params['modelpath'] = save_name.replace('Finetune','Orig') # the pre-trained model to be fine-tuned on
save_name = train_models(data, label, save_name, params)
