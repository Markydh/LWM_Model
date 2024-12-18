# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:13:29 2024

This script generates preprocessed data from wireless communication scenarios_test,
including token generation, patch creation, and data sampling for machine learning models.

@author: salikha4
"""

import numpy as np
import os
from tqdm import tqdm
import time
import pickle
import DeepMIMOv3
import torch

#%% Scenarios List
def scenarios_list():
    """Returns an array of available scenarios_test."""
    return np.array([
        'city_18_denver', 'city_15_indianapolis', 'city_19_oklahoma', 
        'city_12_fortworth', 'city_11_santaclara', 'city_7_sandiego'
    ])

#%% Token Generation
def tokenizer(selected_scenario_names=None, manual_data=None, gen_raw=True):
    """
    Generates tokens by preparing and preprocessing the dataset.

    Args:
        scenario_idxs (list): Indices of the scenarios_test.
        patch_gen (bool): Whether to generate patches. Defaults to True.
        patch_size (int): Size of each patch. Defaults to 16.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data. Defaults to False.
        gen_raw (bool): Whether to generate raw data. Defaults to False.
        save_data (bool): Whether to save the preprocessed data. Defaults to False.
    
    Returns:
        preprocessed_data, sequence_length, element_length: Preprocessed data and related dimensions.
    """

    if manual_data is not None:
        patches = patch_maker(np.expand_dims(np.array(manual_data), axis=1))
        #patches = patch_maker(torch.tensor(manual_data, dtype=torch.complex64).unsqueeze(1))
    else:
        # Patch generation or loading
        deepmimo_data = [DeepMIMO_data_gen(scenario_name) for scenario_name in selected_scenario_names]
        n_scenarios = len(selected_scenario_names)
        
        cleaned_deepmimo_data = [deepmimo_data_cleaning(deepmimo_data[scenario_idx]) for scenario_idx in range(n_scenarios)]
        
        patches = [patch_maker(cleaned_deepmimo_data[scenario_idx]) for scenario_idx in range(n_scenarios)]
        patches = np.vstack(patches) # 把六个数据集的所用用户的信号矩阵叠
   
    # Define dimensions
    # patch_size.shape -> 14840-128-16
    patch_size = patches.shape[2]    # 补丁大小
    n_patches = patches.shape[1]     # 补丁数
    n_masks_half = int(0.15 * n_patches / 2) # 掩码数量 对于每个数据来说 128个补丁包括了实部和虚部
    # sequence_length = n_patches + 1
    # element_length = patch_size 

    # 生成CLS补丁 MASK掩码
    word2id = {'[CLS]': 0.2 * np.ones((patch_size)), '[MASK]': 0.1 * np.ones((patch_size))}

    # Generate preprocessed channels
    preprocessed_data = []
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        sample = make_sample(user_idx, patches, word2id, n_patches, n_masks_half, patch_size, gen_raw=gen_raw)
        preprocessed_data.append(sample)
    # 返回14840个 信道信息  每个信道 分为 128个切片 一个切片大小为16
    return preprocessed_data

#%%
def deepmimo_data_cleaning(deepmimo_data):
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6

#%% Patch Creation
# 返回格式
# (1354, 128, 16)
# (3248, 128, 16)
# (3455, 128, 16)
# (1902, 128, 16)
# (2689, 128, 16)
# (2192, 128, 16)
# 6个数据集，对每个数据集都进行切分补丁 对于数据集一来说 1354个用户 每个用户的H矩阵进行切片，分为实部和虚部，分别是64个切片，切片大小为16
def patch_maker(original_ch, patch_size=16, norm_factor=1e6):
    """
    Creates patches from the dataset based on the scenario.

    Args:-
        patch_size (int): Size of each patch.
        scenario (str): Selected scenario for data generation.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data.
        norm_factor (int): Normalization factor for channels.

    Returns:
        patch (numpy array): Generated patches.
    """
    # idxs = np.where(data['user']['LoS'] != -1)[0]
    
    # # Reshaping and normalizing channels
    # original_ch = data['user']['channel'][idxs]
    flat_channels = original_ch.reshape((original_ch.shape[0], -1)).astype(np.csingle)
    flat_channels_complex = np.hstack((flat_channels.real, flat_channels.imag))
    
    # Create patches
    n_patches = flat_channels_complex.shape[1] // patch_size
    patch = np.zeros((len(flat_channels_complex), n_patches, patch_size))
    for idx in range(n_patches):
        patch[:, idx, :] = flat_channels_complex[:, idx * patch_size:(idx + 1) * patch_size]
    return patch


#%% Data Generation for Scenario Areas
def DeepMIMO_data_gen(scenario):
    """
    Generates or loads data for a given scenario.

    Args:
        scenario (str): Scenario name.
        gen_deepMIMO_data (bool): Whether to generate DeepMIMO data.
        save_data (bool): Whether to save generated data.
    
    Returns:
        data (dict): Loaded or generated data.
    """
    import DeepMIMOv3
    
    parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers = get_parameters(scenario)
    
    # 获取数据集
    deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)
    uniform_idxs = uniform_sampling(deepMIMO_dataset, [1, 1], len(parameters['user_rows']), 
                                    users_per_row=row_column_users[scenario]['n_per_row'])
    data = select_by_idx(deepMIMO_dataset, uniform_idxs)[0]
        
    return data

#%%%
def get_parameters(scenario):
    
    n_ant_bs = 32 #32
    n_ant_ue = 1
    n_subcarriers = 32 #32
    scs = 30e3
        
    row_column_users = {
    'city_18_denver': {
        'n_rows': 85,
        'n_per_row': 82
    },
    'city_15_indianapolis': {
        'n_rows': 80,
        'n_per_row': 79
    },
    'city_19_oklahoma': {
        'n_rows': 82,
        'n_per_row': 75
    },
    'city_12_fortworth': {
        'n_rows': 86,
        'n_per_row': 72
    },
    'city_11_santaclara': {
        'n_rows': 47,
        'n_per_row': 114
    },
    'city_7_sandiego': {
        'n_rows': 71,
        'n_per_row': 83
    }}
    
    parameters = DeepMIMOv3.default_params()
    parameters['dataset_folder'] = '/Users/dianhongyang/Desktop/毕业设计/LWM_MODEL/LWM/scenarios'
    parameters['scenario'] = scenario
    
    if scenario == 'O1_3p5':
        parameters['active_BS'] = np.array([4])
    elif scenario in ['city_18_denver', 'city_15_indianapolis']:
        parameters['active_BS'] = np.array([3])
    else:
        parameters['active_BS'] = np.array([1])
        
    if scenario == 'Boston5G_3p5':
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'][0],
                                            row_column_users[scenario]['n_rows'][1])
    else:
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20
    
    return parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers

#%% Sample Generation
def make_sample(user_idx, patch, word2id, n_patches, n_masks, patch_size, gen_raw=False):
    """
    Generates a sample for each user, including masking and tokenizing.

    Args:
        user_idx (int): Index of the user.
        patch (numpy array): Patches data.
        word2id (dict): Dictionary for special tokens.
        n_patches (int): Number of patches.
        n_masks (int): Number of masks.
        patch_size (int): Size of each patch.
        gen_raw (bool): Whether to generate raw tokens.

    Returns:
        sample (list): Generated sample for the user.
    """
    # 获取 数据集中的 用户的信道矩阵 shape (128,16)
    tokens = patch[user_idx]
    # 在数据集中 加入 [CLS] 标记
    input_ids = np.vstack((word2id['[CLS]'], tokens))
    
    real_tokens_size = int(n_patches / 2)
    masks_pos_real = np.random.choice(range(0, real_tokens_size), size=n_masks, replace=False)
    masks_pos_imag = masks_pos_real + real_tokens_size
    masked_pos = np.hstack((masks_pos_real, masks_pos_imag)) + 1
    
    masked_tokens = []
    for pos in masked_pos:
        original_masked_tokens = input_ids[pos].copy()
        masked_tokens.append(original_masked_tokens)
        if not gen_raw:
            rnd_num = np.random.rand()
            if rnd_num < 0.1:
                input_ids[pos] = np.random.rand(patch_size)
            elif rnd_num < 0.9:
                input_ids[pos] = word2id['[MASK]']
                

    # input_ids：包含 [CLS] 标记和掩码操作后的输入数据，形状为 (129, 16)。
    # masked_tokens：被掩盖补丁的原始值，用于计算模型的重建损失。
    # masked_pos：掩盖补丁的索引，用于指导模型关注这些位置进行预测。
    return [input_ids, masked_tokens, masked_pos]


#%% Sampling and Data Selection
def uniform_sampling(dataset, sampling_div, n_rows, users_per_row):
    """
    Performs uniform sampling on the dataset.

    Args:
        dataset (dict): DeepMIMO dataset.
        sampling_div (list): Step sizes along [x, y] dimensions.
        n_rows (int): Number of rows for user selection.
        users_per_row (int): Number of users per row.

    Returns:
        uniform_idxs (numpy array): Indices of the selected samples.
    """
    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i * users_per_row for i in rows for j in cols])
    
    return uniform_idxs

def select_by_idx(dataset, idxs):
    """
    Selects a subset of the dataset based on the provided indices.

    Args:
        dataset (dict): Dataset to trim.
        idxs (numpy array): Indices of users to select.

    Returns:
        dataset_t (list): Trimmed dataset based on selected indices.
    """
    dataset_t = []  # Trimmed dataset
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]['location'] = dataset[bs_idx]['location']
            dataset_t[bs_idx]['user'] = {k: dataset[bs_idx]['user'][k][idxs] for k in dataset[bs_idx]['user']}
    
    return dataset_t

#%% Save and Load Utilities
def save_var(var, path):
    """
    Saves a variable to a pickle file.

    Args:
        var (object): Variable to be saved.
        path (str): Path to save the file.

    Returns:
        None
    """
    path_full = path if path.endswith('.p') else (path + '.pickle')    
    with open(path_full, 'wb') as handle:
        pickle.dump(var, handle)

def load_var(path):
    """
    Loads a variable from a pickle file.

    Args:
        path (str): Path of the file to load.

    Returns:
        var (object): Loaded variable.
    """
    path_full = path if path.endswith('.p') else (path + '.pickle')
    with open(path_full, 'rb') as handle:
        var = pickle.load(handle)
    
    return var

#%% Label Generation
def label_gen(task, data, scenario, n_beams=64):
    
    idxs = np.where(data['user']['LoS'] != -1)[0]
            
    if task == 'LoS/NLoS Classification':
        label = data['user']['LoS'][idxs]
    elif task == 'Beam Prediction':
        parameters, row_column_users, n_ant_bs, n_ant_ue, n_subcarriers = get_parameters(scenario)
        n_users = len(data['user']['channel'])
        n_subbands = 1
        fov = 120

        # Setup Beamformers
        beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)

        F1 = np.array([steering_vec(parameters['bs_antenna']['shape'], 
                                    phi=azi*np.pi/180, 
                                    kd=2*np.pi*parameters['bs_antenna']['spacing']).squeeze()
                       for azi in beam_angles])

        full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
        for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
            if data['user']['LoS'][ue_idx] == -1:
                full_dbm[:,:,ue_idx] = np.nan
            else:
                chs = F1 @ data['user']['channel'][ue_idx]
                full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
                full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

        best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
        best_beams = best_beams.astype(float)
        best_beams[np.isnan(full_dbm[0,0,:])] = np.nan
        # max_bf_pwr = np.max(np.mean(full_dbm,axis=1), axis=0) 
    
        label = best_beams[idxs]
        
    return label.astype(int)

def steering_vec(array, phi=0, theta=0, kd=np.pi):
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)

def label_prepend(deepmimo_data, preprocessed_chs, task, scenario_idxs, n_beams=64):
    labels = []
    for scenario_idx in scenario_idxs:
        scenario_name = scenarios_list()[scenario_idx]
        # data = DeepMIMO_data_gen(scenario_name)
        data = deepmimo_data[scenario_idx]
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams))
    
    preprocessed_chs = [preprocessed_chs[i] + [labels[i]] for i in range(len(preprocessed_chs))]
    
    return preprocessed_chs

def create_labels(task, scenario_names, n_beams=64):
    labels = []
    for scenario_name in scenario_names:
        data = DeepMIMO_data_gen(scenario_name)
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams))
    return labels
