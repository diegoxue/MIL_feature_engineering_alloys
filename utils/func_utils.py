import numpy as np
from typing import List

def clean_c_f(comp: List[float], features: List[float]) -> float:
    ''' 基于numpy返回comp中非零元素的index '''
    assert len(comp) == len(features)
    comp, features = np.array(comp), np.array(features)
    idx = np.where(comp != 0)
    return comp[idx], features[idx]

def weight_avg(comp: List[float], features: List[float]) -> float:
    assert len(comp) == len(features)
    comp, features = clean_c_f(comp, features)
    return np.dot(comp, features)

def avg_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算加权平均值

        :param comp:
        :param features:
        :param feature_name:
        :return: avg
    '''
    
    comp, features = clean_c_f(comp, features)
    res = weight_avg(features, comp)
    return res

def delta_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算最大值与最小值之差
    '''
    f_avg = avg_func(comp, features)
    comp, features = clean_c_f(comp, features)
    return np.sqrt(weight_avg([(1 - f / f_avg) ** 2 for f in features], comp))

def max_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算最大值
    '''
    
    comp, features = clean_c_f(comp, features)
    return max(features)

def min_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算最小值
    '''
    
    comp, features = clean_c_f(comp, features)
    return min(features)

def range_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算最大值与最小值之差
    '''
    
    comp, features = clean_c_f(comp, features)
    return max(features) - min(features)

def maxc_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        weighted max(feature)
    '''
    
    comp, features = clean_c_f(comp, features)
    idx = np.argmax(features)
    return comp[idx] * features[idx] / avg_func(comp, features)

def minc_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        weighted min(feature)
    '''
    
    comp, features = clean_c_f(comp, features)
    idx = np.argmin(features)
    return comp[idx] * features[idx] / avg_func(comp, features)

def rangec_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        weighted range(feature)
        NOTE: 需要重新review这个函数的实现
    '''
    comp, features = clean_c_f(comp, features)
    idx_min, idx_max = np.argmin(features), np.argmax(features)
    return comp[idx_min] * comp[idx_max] * (features[idx_max] - features[idx_min])


# def std_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
#     '''
#         计算标准差
#     '''
#     
#     return np.std(features)

def medianc_func(comp: List[float], features: List[float], feature_name: str = '') -> float:
    '''
        计算加权中位数
    '''
    comp, features = clean_c_f(comp, features)
    sorted_indices = np.argsort(features)
    sorted_data = features[sorted_indices]
    sorted_weights = comp[sorted_indices]

    cum_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)

    # 找到第一个累计权重大于等于总权重一半的位置
    index = np.searchsorted(cum_weights, total_weight / 2.0)

    if total_weight % 2 == 1 or cum_weights[index] > total_weight / 2.0:
        # 总权重为奇数或者累计权重大于总权重一半
        weighted_median = sorted_data[index]
    else:
        # 总权重为偶数
        weighted_median = (sorted_data[index] + sorted_data[index - 1]) / 2.0

    return weighted_median

def deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    '''
        计算混合物的形成焓
    '''
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    deltaH_sum = 0.
    for i in idxs:
        for j in idxs:
            if i != j:
                deltaH_sum += 4 * comp[i] * comp[j] * mix_e_matrix[i][j]
    return deltaH_sum / 2

def max_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    '''
        任意两个元素形成焓的最大值
    '''
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    max_deltaH = - float('inf')
    for i in idxs:
        for j in idxs:
            if i != j:
                max_deltaH = max(max_deltaH, mix_e_matrix[i][j])
    return max_deltaH

def min_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    '''
        计算混合物的形成焓
    '''
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    min_deltaH = float('inf')
    for i in idxs:
        for j in idxs:
            if i != j:
                min_deltaH = min(min_deltaH, mix_e_matrix[i][j])
    return min_deltaH

def range_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    return max_deltaH_func(comp, mix_e_matrix, elem_names) - min_deltaH_func(comp, mix_e_matrix, elem_names)

def maxc_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    '''
        weighted max(deltaH)
    '''
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    max_deltaH = - float('inf')
    for i in idxs:
        for j in idxs:
            if i != j:
                if max_deltaH < mix_e_matrix[i][j]:
                    max_deltaH = mix_e_matrix[i][j]
                    res = comp[i] * comp[j] * mix_e_matrix[i][j]
    return res

def minc_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    '''
        weighted min(deltaH)
    '''
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    min_deltaH = float('inf')
    for i in idxs:
        for j in idxs:
            if i != j:
                if min_deltaH > mix_e_matrix[i][j]:
                    min_deltaH = mix_e_matrix[i][j]
                    res = comp[i] * comp[j] * mix_e_matrix[i][j]
    return res

def rangec_deltaH_func(comp: List[float], mix_e_matrix: List[List[float]], elem_names: List[str]) -> float:
    assert len(comp) == len(elem_names) == len(mix_e_matrix)
    comp = np.array(comp)
    idxs = np.where(comp != 0)[0]
    min_deltaH = float('inf')
    max_deltaH = - float('inf')
    for i in idxs:
        for j in idxs:
            if i != j:
                if min_deltaH > mix_e_matrix[i][j]:
                    min_deltaH = mix_e_matrix[i][j]
                    min_comp = comp[i] * comp[j]
                if max_deltaH < mix_e_matrix[i][j]:
                    max_deltaH = mix_e_matrix[i][j]
                    max_comp = comp[i] * comp[j]
    return min_comp * max_comp * (max_deltaH - min_deltaH)

def dr_func(comp: List[float], r_list: List[float]) -> float:
    comp, r_list = clean_c_f(comp, r_list)
    res = 0.
    for i in range(len(comp)):
        for j in range(len(comp)):
            if i != j:
                res += comp[i] * comp[j] * abs(r_list[i] - r_list[j])
    return res / 2

def ita_func(comp: List[float], g_list: List[float]) -> float:
    comp, g_list = clean_c_f(comp, g_list)
    g_avg = avg_func(comp, g_list)
    res = 0.
    for i in range(len(comp)):
        tmp = comp[i] * 2 * (g_list[i] - g_avg) / (g_list[i] + g_avg)
        if tmp:
            res += tmp / (1 + 0.5  * abs(tmp))
    return res

def mu_func(comp: List[float], e_list: List[float], r_list: List[float]) -> float:
    r_delta = delta_func(comp, r_list)
    comp, e_list = clean_c_f(comp, e_list)  # NOTE: maybe bug
    res = 0.
    for i in range(len(comp)):
        res += comp[i] * e_list[i] * r_delta / 2
    return res

def A_func(comp: List[float], e_list: List[float], r_list: List[float], g_list) -> float:
    mu = mu_func(comp, e_list, r_list)
    g_avg = avg_func(comp, g_list)
    r_delta = delta_func(comp, r_list)
    return g_avg * r_delta * (1 + mu) / (1 - mu)

def F_func(comp: List[float], e_list: List[float], r_list: List[float], g_list) -> float:
    mu = mu_func(comp, e_list, r_list)
    g_avg = avg_func(comp, g_list)
    return 2 * g_avg / (1 - mu)

def w6_func(comp: List[float], w_list: List[float]) -> float:
    comp, w_list = clean_c_f(comp, w_list)
    w = avg_func(comp, w_list)
    return w ** 6

def deltaS_mix_func(comp: List[float]) -> float:
    comp = np.array(comp)
    comp = comp[np.where(comp != 0)]
    return sum([-8.314 * c * np.log(c) for c in comp])

def LAMBDA_func(comp: List[float], r_list: List[float]) -> float:
    comp, r_list = clean_c_f(comp, r_list)
    r_delta = delta_func(comp, r_list)
    return deltaS_mix_func(comp) / r_delta ** 2

def gamma_func(comp: List[float], r_list: List[float]) -> float:
    comp, r_list = clean_c_f(comp, r_list)
    r_avg = avg_func(comp, r_list)
    r_min, r_max = min(r_list), max(r_list)
    res = (1 - np.sqrt(((r_min + r_avg) ** 2 - r_avg ** 2) / (r_min + r_avg) ** 2))\
            / (1 - np.sqrt(((r_max + r_avg) ** 2 - r_avg ** 2) / (r_max + r_avg) ** 2))
    return res

def strain_synth_f1_func(comp: List[float]) -> float:
    ''' 
        Synthetic feature #1 for strain.
        Synthetic features are statistically determined by Dr. ZhangYan's methods.
        They represents (in some sense) rectified conventional features 
        based on experimental data.

        Order: ['Al', 'Si', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']
    '''
    f1_list = np.array([0.95, 0.84, 0.03, 0.33, 0.77, 0.13, 0.18, 0.8, 1, 0.43, 0.39, 0.15, 0.96, 0.53, 0.8, 0.75])
    return avg_func(comp, f1_list)

def strain_synth_f2_func(comp: List[float]) -> float:
    ''' Synthetic feature #2 for strain'''
    f2_list = np.array([0.85, 0.25, 0.83, 0.52, 0.83, 0.96, 0.05, 0.68, 0.99, 0.13, 0.05, 0.07, 0.67, 0.02, 0.04, 0.59])
    return avg_func(comp, f2_list)

def strength_synth_f1_func(comp: List[float]) -> float:
    ''' Synthetic feature #1 for strength '''
    f1_list = np.array([0.46, 0.15, 0.99, 0.26, 0.71, 0.98, 0.31, 0.47, 0.38, 0.77, 0.34, 0.48, 0.41, 0.93, 0.16, 0.93])
    return avg_func(comp, f1_list)

def strength_synth_f2_func(comp: List[float]) -> float:
    ''' Synthetic feature #2 for strength '''
    f2_list = np.array([0.83, 0.63, 0.02, 0.05, 0.1, 0.15, 0.19, 0.21, 0.1, 0.58, 0.01, 0.23, 0.29, 0.7, 0.82, 0.41])
    return avg_func(comp, f2_list)

if __name__ == '__main__':
    test_comp = [0, 0, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0, 0, 0]
    assert round(strain_synth_f1_func(test_comp), 4) == 0.3675
    assert round(strain_synth_f2_func(test_comp), 4) == 0.5225
    assert round(strength_synth_f1_func(test_comp), 4) == 0.535
    assert round(strength_synth_f2_func(test_comp), 4) == 0.1475