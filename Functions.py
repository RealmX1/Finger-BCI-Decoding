from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import tensorflow as tf

import scipy
import scipy.io
from scipy.signal import resample

from sklearn.utils.class_weight import compute_class_weight

import os

# ============ 配置常量 (Configuration Constants) ============
# 滑动窗口步长 (Sliding window step size)
# 论文: "step size of 125 ms" @ 1024Hz = 128 samples
DEFAULT_STEP_SIZE = 128

# EEGNet-specific imports
from EEGModels_tf import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from utils.training_callbacks import TableEpochLogger

# 注意：为确保实验可复现性，建议在 params 中设置 'random_seed' 参数
# Note: To ensure reproducibility, it's recommended to set 'random_seed' in params


###### data segmenting and relabeling functions ######
def segment_data(
    data: np.ndarray,
    labels: np.ndarray,
    segment_size: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size and step_size must be positive.")

    num_trials, num_channels, num_samples = data.shape
    segments = []

    for start in range(0, num_samples - segment_size + 1, step_size):
        end = start + segment_size
        segments.append(data[:, :, start:end])

    segmented_data = np.concatenate(segments, axis=0)
    # repeat labels 
    repeated_labels = np.tile(labels, len(segments))
    trial_indices = range(num_trials)
    repeated_indices = np.tile(trial_indices, len(segments))

    repeated_labels = repeated_labels[~np.isnan(segmented_data).any(axis=(1,2))]
    repeated_indices = repeated_indices[~np.isnan(segmented_data).any(axis=(1,2))]
    segmented_data = segmented_data[~np.isnan(segmented_data).any(axis=(1,2)),:,:]

    return segmented_data, repeated_labels, repeated_indices


def filter_and_relabel(
    data: np.ndarray,
    label: np.ndarray,
    keep_labels: List[int],
    new_labels: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    filtered_label = label[np.isin(label,keep_labels)]
    filtered_data = data[np.isin(label,keep_labels)]
    filtered_label = np.array([new_labels[l] for l in filtered_label])
    return filtered_data, filtered_label

def generate_paths(
    subj_id: int,
    task: str,
    nclass: int,
    session_num: int,
    model_type: str,
    data_folder: str
) -> List[str]:
    """
    生成训练数据路径。

    对于 Base (Orig) 模型：
        - 离线数据 (OfflineImagery/OfflineMovement)
        - 之前所有session的在线数据 (Base + Finetune)

    对于 Finetune 模型：
        - 当前session的Base测试数据

    参数:
        subj_id: 被试ID (1-21)
        task: 'MI' (运动想象) 或 'ME' (运动执行)
        nclass: 类别数 (2 或 3)
        session_num: session编号 (1-5)
        model_type: 'Orig' (Base模型) 或 'Finetune' (微调模型)
        data_folder: 数据根目录

    返回:
        data_paths: 数据目录路径列表
    """
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')
    task_suffix = 'Imagery' if task == 'MI' else 'Movement'
    data_paths = []

    if model_type == 'Orig':  # Base模型：累积训练
        # 1. 离线数据
        offline_dir = os.path.join(subject_folder, f'Offline{task_suffix}')
        if os.path.exists(offline_dir):
            data_paths.append(offline_dir)

        # 2. 之前session的在线数据（Base + Finetune）
        for prev_session in range(1, session_num):
            for data_type in ['Base', 'Finetune']:
                online_dir = os.path.join(
                    subject_folder,
                    f'Online{task_suffix}_Sess{prev_session:02}_{nclass}class_{data_type}'
                )
                if os.path.exists(online_dir):
                    data_paths.append(online_dir)

    elif model_type == 'Finetune':  # 微调模型：仅当前session的Base数据
        finetune_dir = os.path.join(
            subject_folder,
            f'Online{task_suffix}_Sess{session_num:02}_{nclass}class_Base'
        )
        if os.path.exists(finetune_dir):
            data_paths.append(finetune_dir)

    return data_paths


def generate_test_paths(
    subj_id: int,
    task: str,
    nclass: int,
    session_num: int,
    model_type: str,
    data_folder: str
) -> Optional[str]:
    """
    生成测试数据路径。

    测试数据逻辑：
        - Base模型测试：当前session的Base数据 (如 OnlineImagery_Sess01_2class_Base)
        - Finetune模型测试：当前session的Finetune数据 (如 OnlineImagery_Sess01_2class_Finetune)

    参数:
        subj_id: 被试ID (1-21)
        task: 'MI' (运动想象) 或 'ME' (运动执行)
        nclass: 类别数 (2 或 3)
        session_num: session编号 (1-5)
        model_type: 'Orig' (测试Base模型) 或 'Finetune' (测试Finetune模型)
        data_folder: 数据根目录

    返回:
        test_path: 测试数据目录路径，若不存在则返回None
    """
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')
    task_suffix = 'Imagery' if task == 'MI' else 'Movement'

    # 测试数据类型与模型类型相同
    test_type = 'Base' if model_type == 'Orig' else 'Finetune'

    test_dir = os.path.join(
        subject_folder,
        f'Online{task_suffix}_Sess{session_num:02}_{nclass}class_{test_type}'
    )

    return test_dir if os.path.exists(test_dir) else None


def load_and_filter_data(
    data_paths: List[str],
    params: Dict[str, Union[int, float, str, List[float]]]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float, str, List[float]]]]:
    # H-3: 添加空路径检查
    if not data_paths:
        raise ValueError("data_paths is empty. No data paths provided for loading.")

    # 检查路径是否存在
    for path in data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path does not exist: {path}")

    label = [] #nTrials
    data = []#nTrials, nChannels, nSamples

    if params['nclass'] == 2:
        keep_labels = [1,4] # thumb; pinky
        new_labels = {1: 1, 4: 2}
    elif params['nclass'] == 3:
        keep_labels = [1,2,4] # thumb; index; pinky
        new_labels = {1: 1, 2: 2, 4: 3}
    else:
        raise ValueError("nclass must be either 2 or 3.")

    total_files = 0
    for filepath in data_paths:
        files = sorted(os.listdir(filepath))
        print(f"加载 {len(files)} 个文件从: {os.path.basename(filepath)}")
        for filename in files:
            cur_data = []
            file_path = os.path.join(filepath, filename)
            total_files += 1

            mat = scipy.io.loadmat(file_path)

            # M-10: 验证必需字段
            if 'eeg' not in mat:
                raise KeyError(f"Missing 'eeg' field in {file_path}")
            if 'event' not in mat:
                raise KeyError(f"Missing 'event' field in {file_path}")

            eeg = mat['eeg']
            event = mat['event']

            signals = eeg['data'][0][0]
            params['srate'] = eeg['fsample'][0][0][0][0]
            start_idx, end_idx, target = [], [], []

            # Iterate through events
            for i in range(event.shape[1]):
                evt = event[0, i]
                event_type = evt['type'][0]
                sample = evt['sample'][0][0]
                value = evt['value'][0][0]

                if event_type == 'Target':
                    start_idx.append(sample-1) # 0-index
                    target.append(value)
                elif event_type == 'TrialEnd':
                    end_idx.append(sample-1) # 0-index

            cur_label = target

            for i in range(len(start_idx)):
                tmp = signals[:,int(start_idx[i]):int(end_idx[i])]
                tmp = tmp[:,:min(np.size(tmp,1), int(params['maxtriallen']*params['srate']))]
                tmp = np.pad(tmp,((0,0),(0,int(params['maxtriallen']*params['srate'])-np.size(tmp,1))), 'constant', constant_values=np.nan)
        
                cur_data.append(tmp)
            cur_data = np.array(cur_data)

            # CAR
            cur_data = cur_data-cur_data.mean(axis=1, keepdims=True)
            
            data.append(cur_data)
            label.append(cur_label) #nTrials

    #### Preprocessing ####

    data = np.concatenate(data,axis=0)
    label = np.concatenate(label,axis=0)
    label = label.flatten()
    print(f"数据加载完成: {total_files} 个文件, {data.shape[0]} trials, {data.shape[1]} 通道")

    # relabel the data
    data, label = filter_and_relabel(data, label, keep_labels, new_labels)
    return data, label, params

def train_models(
    data: np.ndarray,
    label: np.ndarray,
    save_name: str,
    params: Dict[str, Union[int, float, str, List[float]]]
) -> str:

    if 'modelpath' in params.keys(): # finetune
        print(f'Fine-tuning model: {save_name}...')
    else:
        print(f'Training model: {save_name}...')
    K.set_image_data_format('channels_last')

    # 设置随机种子以确保可复现性 (Set random seed for reproducibility)
    random_seed = params.get('random_seed', 42)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    print(f'Using random seed: {random_seed}')

    nTrial = len(data)
    nChan = np.size(data,axis=1)
    shuffled_idx = np.random.permutation(nTrial)

    # split into training/validation sets
    train_percent = 0.8
    train_idx = range(int(train_percent*nTrial))
    train_idx = shuffled_idx[train_idx]
    val_idx = np.setdiff1d(shuffled_idx,train_idx)
    X_train = data[train_idx,:,:]
    X_validate = data[val_idx,:,:]
    Y_train = label[train_idx]
    Y_validate = label[val_idx]

    ############################# preprocessing ##################################
    # segment data
    DesiredLen = int(params['windowlen']*params['downsrate'])

    segment_size = int(params['windowlen']*params['srate'])  # size of each segment - 1 s
    step_size = params.get('step_size', DEFAULT_STEP_SIZE)
    X_train, Y_train, I_train = segment_data(X_train, Y_train, segment_size, step_size)
    X_validate, Y_validate, I_validate = segment_data(X_validate, Y_validate, segment_size, step_size)

    # downsample
    X_train = resample(X_train, DesiredLen, t=None, axis=2, window=None, domain='time')
    X_validate = resample(X_validate, DesiredLen, t=None, axis=2, window=None, domain='time')

    # bandpass filtering
    padding_length = 100  # Number of zeros to pad
    padded_train = np.pad(X_train, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)
    padded_validate = np.pad(X_validate, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)

    b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass', fs=params['downsrate'])
    X_train = scipy.signal.lfilter(b, a, padded_train, axis=-1)
    X_validate = scipy.signal.lfilter(b, a, padded_validate, axis=-1)

    X_train = X_train[:,:,padding_length:-padding_length]
    X_validate = X_validate[:,:,padding_length:-padding_length]

    # zscore
    X_train = scipy.stats.zscore(X_train, axis=2, nan_policy='omit')
    X_validate = scipy.stats.zscore(X_validate, axis=2, nan_policy='omit')
        
    ############################# EEGNet portion ##################################
    kernels, chans, samples = 1, nChan, DesiredLen
    batch_size, epochs = 16, 300
    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)

    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    if 'modelpath' in params.keys(): # finetune: larger dropout ratio
        params['dropout_ratio'] = 0.65
    else:
        params['dropout_ratio'] = 0.5
    model = EEGNet(nb_classes = params['nclass'], Chans = chans, Samples = samples, 
                dropoutRate = params['dropout_ratio'], kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')  
    
    model.summary()

    # Callbacks
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    if 'modelpath' in params.keys(): # finetune: smaller starting lr
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                metrics = ['accuracy'])
   
    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=save_name, verbose=1,monitor='val_accuracy',
                                    mode='max',save_best_only=True)

    # 动态计算类别权重，处理类别不平衡
    train_labels = np.argmax(Y_train, axis=1)
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}
    print(f"Class weights: {class_weights}")

    if 'modelpath' in params.keys(): # finetune
        params['epochs'] = 100
        params['layers_to_freeze'] = 4
        model.load_weights(params['modelpath'])
        model.trainable = True

        for model_layer in model.layers[:params['layers_to_freeze']]:
            print(f"FREEZING LAYER: {model_layer}")
            model_layer.trainable = False

    else:
        params['epochs'] = 300

    # 使用自定义表格日志回调 (Use custom table logger callback)
    epoch_logger = TableEpochLogger(header_every=20, keep_every=5)

    model.fit(X_train, Y_train, batch_size = batch_size, epochs = params['epochs'],
                verbose = 0, validation_data=(X_validate, Y_validate),
                callbacks=[checkpointer, callback_es, callback_lr, epoch_logger],
                class_weight = class_weights)

    print("Training Finished!")
    print(f"Model saved to {save_name}")
    return save_name
