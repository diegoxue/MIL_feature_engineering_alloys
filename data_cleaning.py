import concurrent.futures
import itertools
import logging
import math
import pickle
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import combinations
from threading import Lock, Thread
from typing import List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

from func_utils import *

# 删除含有nan的行，删除含有delete_rows_with_strs中的str的行，删除含有delete_strs中的str的列，将剩下的含有str的列的str替换为0，将剩下的含有str的列的str转换为float
def initial_clean(base_data: pd.DataFrame, property_name: str, delete_rows_with_strs: List[str] = [], delete_strs: List[str] = []):
    '''
    1. 删除含有nan的行
    2. 删除含有delete_rows_with_strs中的str的行
    3. 删除含有delete_strs中的str的列
    4. 将剩下的含有str的列的str替换为0
    5. 将剩下的含有str的列的str转换为float

    :param base_data:
    :param property_name:
    :param delete_rows_with_strs:
    :param delete_strs:

    :return: single_prop_data
    '''
    single_prop_data = (base_data[ELEM_NAMES + [property_name]]).copy()
    single_prop_data.dropna(subset = [property_name], inplace = True)
    if delete_rows_with_strs :
        # print(f'delete {len(base_data) - len(single_prop_data)} rows with nan, for property {property_name}')
        mask = single_prop_data[property_name].apply(
            lambda x: all([c not in x for c in delete_rows_with_strs]) if isinstance(x, str) else True)
        # print(f'delete {sum(mask.apply(lambda x: not x))} rows with strs {delete_rows_with_strs}, for property {property_name}')
        single_prop_data = single_prop_data[mask]
    ''' 删除property_name列中含有delete_strs中的str,保留剩下的字符串或者数字, for STRAIN '''
    if delete_strs : single_prop_data[property_name] = single_prop_data[property_name].apply(
        lambda x: re.sub('|'.join(delete_strs), '', x) if isinstance(x, str) else x)
    single_prop_data.fillna(0., inplace = True)
    single_prop_data = single_prop_data.astype(float)
    return single_prop_data

def re_distribute(property_val: NDArray, source_val: float, dest_vals: Tuple[float, float]) -> NDArray:
    ''' 将property_val中的source_val值重新分配到上下限为dest_vals的uniform distribution中 '''
    assert len(dest_vals) == 2
    index = np.where(property_val == source_val)
    np.random.seed(42)
    property_val[index] = np.round(np.random.uniform(dest_vals[0], dest_vals[1], len(index[0])), 2)
    return property_val

# 对所有的行元素成分进行归一化，并剔除重复元素成分的行，误差限0.1%，保存被剔除的行至另一个列表, keep参数只能是'zhang', 'wang'或者'mean'
def normalize_compositions(data_comp_prop: pd.DataFrame, elem_names: List[str], property_name: str, keep: str = 'zhang'):
    '''
    请帮我完成功能描述
    1 对所有的行元素成分进行归一化
    2 剔除重复元素成分的行, 误差限0.1%
    3 保存被剔除的行至另一个列表(optional)

    :param data_comp_prop:
    :param elem_names:
    :param  keep: 'zhang', 'wang' or'mean'
    :return: data_comp_prop
    '''
    data_comp_prop = data_comp_prop.copy()
    data_comp_prop[elem_names] = data_comp_prop[elem_names].apply(lambda x: x / x.sum(), axis = 1)
    data_comp_prop = data_comp_prop.round(3)
    data_comp_prop = data_comp_prop.drop_duplicates()   # 167 rows for strength
    '''Function 3, save the dropped rows to another list(optional)'''
    # duplicates = data_comp_prop[data_comp_prop.duplicated(subset = elem_names, keep = False)]
    # duplicates.to_excel('Data&Code\\RemergeData_20231222\\duplicates.xlsx')
    keep_list = ['zhang', 'wang', 'mean']
    assert keep in keep_list
    keep_index = keep_list.index(keep)
    if keep_index <= 1:
        data_comp_prop = data_comp_prop.drop_duplicates(subset = elem_names, keep = ['first', 'last'][keep_index])
    else:
        # 计算以elem_names为分组的property_name的均值
        keys = data_comp_prop[elem_names].apply(lambda x: '_'.join([str(i) for i in x]), axis = 1).tolist()
        values = data_comp_prop[property_name].tolist()
        buffer = {}
        for k, v in zip(keys, values):
            if k not in buffer.keys(): buffer[k] = []
            buffer[k].append(v)
        ordered_key_set = []
        for k in keys:
            if k not in ordered_key_set: ordered_key_set.append(k)
        mean_props = []
        for k in ordered_key_set:
            mean_props.append(np.mean(buffer[k]))
        data_comp_prop = data_comp_prop[elem_names].drop_duplicates(subset = elem_names, keep = 'first')
        data_comp_prop[property_name] = mean_props
    return data_comp_prop


# 读取原子特征,从Data&Code\RemergeData_20231222\From_ZhangYan_to_Xian_20231222\ELEMENTS16.csv
def read_atomic_features(elem_names: List[str], atomic_features_file_path: str):
    '''
    读取原子特征

    :param elem_names:
    :param atomic_features_file_path:
    :return: atomic_features
    '''
    atomic_features = pd.read_csv(atomic_features_file_path, index_col = 0)
    # 确认atomic_features列名元素与elem_names一致
    assert set(atomic_features.columns) == set(elem_names)
    # 调整atomic_features列名元素顺序与elem_names一致
    return atomic_features[elem_names]

def clean_z_score(data_comp_prop: pd.DataFrame, property_name: str):
    '''
        Z-Score方法剔除异常值
        1. 对每一列特征进行Z-Score标准化
        2. 计算每一行的Z-Score和
        3. 剔除Z-Score和大于3的行
    '''
    while True:
        data_comp_prop = data_comp_prop.copy()
        std_v = data_comp_prop[property_name].std()
        mean_v = data_comp_prop[property_name].mean()
        outliers_idx = data_comp_prop[property_name].apply(lambda x: abs((x - mean_v) / std_v) >= 3)
        if sum(outliers_idx): print(f'Zscore clean {sum(outliers_idx)} outliers, for property {property_name}')
        data_comp_prop = data_comp_prop[~ outliers_idx]
        if sum(outliers_idx) == 0: break
    return data_comp_prop

def clean_isolation_forest(data_comp_prop: pd.DataFrame, property_name: str, contamination_ratio: float = 0.025):
    '''
        Isolation Forest方法剔除异常值
    '''
    model = IsolationForest(contamination = contamination_ratio, random_state = 13)
    all_data = data_comp_prop
    model.fit(all_data)
    outliers = model.predict(all_data) == -1
    print(f'Isolation forest clean {sum(outliers)} outliers, for property {property_name}')
    return data_comp_prop[~ outliers]

def clean_local_outlier(data_comp_prop: pd.DataFrame, property_name: str, contamination_ratio: float = 0.025):
    '''
        Local Outlier Factor方法剔除异常值
    '''
    model = LocalOutlierFactor(contamination = contamination_ratio, n_neighbors = 20)
    all_data = data_comp_prop
    outliers = model.fit_predict(all_data) == -1
    print(f'Local Outlier Factor clean {sum(outliers)} outliers, for property {property_name}')
    return data_comp_prop[~ outliers]

def clean_oneclass_svm(data_comp_prop: pd.DataFrame, property_name: str, contamination_ratio: float = 0.025):
    '''
        One Class SVM方法剔除异常值
    '''
    model = OneClassSVM(nu = contamination_ratio)
    all_data = data_comp_prop
    model.fit(all_data)
    outliers = model.predict(all_data) == -1
    print(f'One Class SVM clean {sum(outliers)} outliers, for property {property_name}')
    return data_comp_prop[~ outliers]

def clean_pca(data_comp_prop: pd.DataFrame, property_name: str):
    ''' 
        PCA方法剔除异常值
    '''
    pca = PCA(n_components = 2)
    all_data = data_comp_prop
    data_pca = pca.fit_transform(all_data)

    # 计算椭球体参数
    cov = np.cov(data_pca.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    radii = np.sqrt(5.991 * eig_vals)

    # 检测异常数据点
    dist = np.sum((data_pca @ eig_vecs / radii)**2, axis = 1)
    outliers = dist > 1
    print(f'clean {sum(outliers)} outliers, for property {property_name}, in PCA')

    # 输出异常数据点
    return data_comp_prop[~ outliers]

''' 2023.12.24 implementation '''
# data_comp_prop = clean_z_score(data_comp_prop, selected_property_name)
# data_comp_prop = clean_isolation_forest(data_comp_prop, selected_property_name)
# data_comp_prop = clean_local_outlier(data_comp_prop, selected_property_name)
# data_comp_prop = clean_oneclass_svm(data_comp_prop, selected_property_name)
# data_comp_prop = clean_pca(data_comp_prop, selected_property_name)

def read_mix_enthalpies(elem_names: List[str], mix_enthalpies_path: str) -> dict:
    '''
        读取混合焓数据
    '''
    mix_e_df = pd.read_csv(mix_enthalpies_path)
    mix_e_data = mix_e_df.to_numpy().flatten().tolist()
    mix_e_data_iter = iter(mix_e_data)
    mix_e_keys = mix_e_df.columns.values.tolist()
    tmp_keys = []
    ''' 创建一个键值为elem_names的字典,值为空字典 '''
    mix_e_dict = {}
    for e in elem_names: mix_e_dict[e] = {}
    for i in range(len(elem_names)):
        for j in range(i+1, len(elem_names)):
            e_i, e_j = elem_names[i], elem_names[j]
            if i + 1 == j:
                tmp_keys.append(e_i + e_j)
            else:
                tmp_keys.append(e_j)
            mix_e = next(mix_e_data_iter)
            mix_e_dict[e_i][e_j] =  mix_e_dict[e_j][e_i] = mix_e
    for k1, k2 in zip(tmp_keys, mix_e_keys):
        assert k1 in k2
    assert mix_e_dict['Al']['Si'] == -19 and mix_e_dict['Cr']['Ni'] == -7 and mix_e_dict['V']['Mo'] == 0
    ''' 创建混合焓矩阵 '''
    mix_e_matrix = np.zeros(shape = (len(elem_names), len(elem_names)), dtype = float)
    for i in range(len(elem_names)):
        for j in range(i+1, len(elem_names)):
            e_i, e_j = elem_names[i], elem_names[j]
            mix_e_matrix[i][j] = mix_e_matrix[j][i] = mix_e_dict[e_i][e_j]
    return mix_e_dict, mix_e_matrix

def cal_primitive_features(data_comp_prop: pd.DataFrame, elem_names: List[str], atomic_features: pd.DataFrame):
    ''' 计算初级原子特征 '''
    data_comp_prop = data_comp_prop.copy()
    atomic_features_names = atomic_features.index.values.tolist()   # 1 row, 1 composition, row_index: 1 atomic feature
    all_comps = data_comp_prop[elem_names].values.tolist()
    func_list = [avg_func, delta_func, max_func, min_func, range_func, maxc_func, minc_func, rangec_func]
    func_names = ['avg', 'delta', 'max', 'min', 'range', 'maxc', 'minc', 'rangec']
    column_name_buffer = []
    column_val_buffer = []
    for a_name in atomic_features_names:
        for f, f_name in zip(func_list, func_names):
            a_feature = atomic_features.loc[a_name].values.tolist()
            column_name_buffer.append( f_name + '.' + a_name )
            ''' 遍历data_comp_prop的每一行,计算元素的原子特征 '''
            column_val_buffer.append( [f(comp, a_feature) for comp in all_comps] )
    ''' 多列一次性添加到data_comp_prop '''
    _tmp = pd.DataFrame(np.array(column_val_buffer).T, columns = column_name_buffer)
    data_comp_prop = data_comp_prop.reset_index(drop = True)
    data_comp_prop = pd.concat([data_comp_prop, _tmp], axis = 1)

    # for name, val in zip(data_comp_prop.columns.values.tolist(), data_comp_prop.loc[0].values.tolist()): print(name, val)
    return data_comp_prop

def cal_H_features(data_comp_prop: pd.DataFrame, elem_names: List[str], mix_e_matrix: List[List[float]]):
    all_comps = data_comp_prop[elem_names].copy().values.tolist()
    func_list = [deltaH_func, max_deltaH_func, min_deltaH_func, range_deltaH_func, maxc_deltaH_func, minc_deltaH_func, rangec_deltaH_func]
    func_names = ['deltaH', 'max_deltaH', 'min_deltaH', 'range_deltaH', 'maxc_deltaH', 'minc_deltaH', 'rangec_deltaH']
    column_name_buffer = []
    column_val_buffer = []
    for f, f_name in zip(func_list, func_names):
        ''' 遍历data_comp_prop的每一行,计算元素的原子特征 '''
        column_name_buffer.append( f_name )
        column_val_buffer.append( [f(comp, mix_e_matrix, elem_names) for comp in all_comps] )
    ''' 多列一次性添加到data_comp_prop '''
    _tmp = pd.DataFrame(np.array(column_val_buffer).T, columns = column_name_buffer)
    data_comp_prop = data_comp_prop.reset_index(drop = True)
    data_comp_prop = pd.concat([data_comp_prop, _tmp], axis = 1)

    return data_comp_prop

def cal_complex_features(data_comp_prop: pd.DataFrame, elem_names: List[str], \
                        atomic_features: pd.DataFrame, mix_e_matrix: List[List[float]]):
    all_comps = data_comp_prop[elem_names].copy().values.tolist()
    r_list = atomic_features.loc['R'].values
    e_list = atomic_features.loc['E'].values
    g_list = atomic_features.loc['G'].values
    w_list = atomic_features.loc['WF'].values

    column_name_buffer = []
    column_val_buffer = []

    column_name_buffer.append('D.R.')
    column_val_buffer.append([dr_func(comp, r_list) for comp in all_comps])

    column_name_buffer.append('ita')
    column_val_buffer.append([ita_func(comp, g_list) for comp in all_comps])

    column_name_buffer.append('mu')
    column_val_buffer.append([mu_func(comp, e_list, r_list) for comp in all_comps])

    column_name_buffer.append('A')
    column_val_buffer.append([A_func(comp, e_list, r_list, g_list) for comp in all_comps])

    column_name_buffer.append('F')
    column_val_buffer.append([F_func(comp, e_list, r_list, g_list) for comp in all_comps])

    column_name_buffer.append('w6')
    column_val_buffer.append([w6_func(comp, w_list) for comp in all_comps])

    column_name_buffer.append('deltaS_mix')
    column_val_buffer.append([deltaS_mix_func(comp) for comp in all_comps])

    column_name_buffer.append('LAMBDA')
    column_val_buffer.append([LAMBDA_func(comp, r_list) for comp in all_comps])

    column_name_buffer.append('gamma')
    column_val_buffer.append([gamma_func(comp, r_list) for comp in all_comps])

    _tmp = pd.DataFrame(np.array(column_val_buffer).T, columns = column_name_buffer)
    # data_comp_prop = data_comp_prop.reset_index()
    data_comp_prop = pd.concat([data_comp_prop, _tmp], axis = 1)

    return data_comp_prop

def cal_synthetic_features(data_comp_prop: pd.DataFrame, elem_names: List[str], property_str: str):
    all_comps = data_comp_prop[elem_names].copy().values.tolist()

    column_name_buffer = []
    column_val_buffer = []

    if 'strength' in property_str.lower():
        column_name_buffer.append('strength_synth_f1')
        column_val_buffer.append([strength_synth_f1_func(comp) for comp in all_comps])

        column_name_buffer.append('strength_synth_f2')
        column_val_buffer.append([strength_synth_f2_func(comp) for comp in all_comps])
    else:
        ''' strain '''
        column_name_buffer.append('strain_synth_f1')
        column_val_buffer.append([strain_synth_f1_func(comp) for comp in all_comps])

        column_name_buffer.append('strain_synth_f2')
        column_val_buffer.append([strain_synth_f2_func(comp) for comp in all_comps])

    _tmp = pd.DataFrame(np.array(column_val_buffer).T, columns = column_name_buffer)
    data_comp_prop = pd.concat([data_comp_prop, _tmp], axis = 1)

    return data_comp_prop

if __name__ == '__main__':
    use_strain = True   # user parameter

    '''  ---------------DO NOT CHANGE CODES BELOW--------------- '''
    PROPERTY_NAME_STRENGTH, PROPERTY_NAME_STRAIN = 'σmax (MPa)', 'εf (%)'
    if use_strain:
        selected_property_name, property_str = PROPERTY_NAME_STRAIN, 'strain'
    else:
        selected_property_name, property_str = PROPERTY_NAME_STRENGTH, 'strength'

    xlsx_file_path = '高熵合金数据_checked_20240107.xlsx'
    ELEM_NAMES = ['Al', 'Si', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']

    ''' 读取数据 '''
    base_data = pd.read_excel(xlsx_file_path) # 189 rows
        
    if use_strain:
        ''' 希望利用strain数据中<部分>应变30或50未压断的数据 '''
        data_comp_prop = initial_clean(base_data, selected_property_name, delete_strs = ['>', '\uff1e'])
    else:
        ''' 强度模型中不利用任何未压断的数据(包含>符号或者\uff1e的数据) '''
        data_comp_prop = initial_clean(base_data, selected_property_name, delete_rows_with_strs = ['>', '\uff1e'])  # 178 rows

    # 针对含有重复元素成分的行，默认性能数据取平均值
    data_comp_prop = normalize_compositions(data_comp_prop, ELEM_NAMES, selected_property_name, keep = 'mean')  # 157 rows
    # print(data_comp_prop.loc[98]), exit() # should be 1793.50

    ''' 原子特征 '''
    atomic_features = read_atomic_features(ELEM_NAMES, 'ELEMENTS16.csv')

    _, mix_e_matrix = read_mix_enthalpies(ELEM_NAMES, 'H-16.csv')

    data_comp_prop = cal_primitive_features(data_comp_prop, ELEM_NAMES, atomic_features)

    data_comp_prop = cal_H_features(data_comp_prop, ELEM_NAMES, mix_e_matrix)

    data_comp_prop = cal_complex_features(data_comp_prop, ELEM_NAMES, atomic_features, mix_e_matrix)

    ''' 合成特征, ref Zhang Yan, PhD thesis, 2023 '''
    # data_comp_prop = cal_synthetic_features(data_comp_prop, ELEM_NAMES, property_str)

    if use_strain:
        ''' 
            strain数据中有大量应变30或50未压断的数据, 需要重新分配到30~32.5和50~52.5之间, 保证数据分布均匀, 模型训练时稳定
        '''
        data_comp_prop[selected_property_name] = re_distribute(data_comp_prop[selected_property_name], 50., (50, 52.5))
        data_comp_prop[selected_property_name] = re_distribute(data_comp_prop[selected_property_name], 30., (30, 32.5))

    ''' clean data after feature construction '''
    data_comp_prop = clean_z_score(data_comp_prop, selected_property_name)
    data_comp_prop = clean_isolation_forest(data_comp_prop, selected_property_name)
    data_comp_prop = clean_local_outlier(data_comp_prop, selected_property_name)
    data_comp_prop = clean_oneclass_svm(data_comp_prop, selected_property_name)
    data_comp_prop = clean_pca(data_comp_prop, selected_property_name)

    ''' save data '''
    data_comp_prop.reset_index(drop = True).to_excel(f'{property_str.lower()}_cleaned_test.xlsx')