import random
from typing import Tuple
import deprecated
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import uuid
from numpy.typing import NDArray
from sklearn import ensemble, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              RationalQuadratic, WhiteKernel)
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin, TransformerMixin

from func_utils import get_random_forest, get_etr

def custom_cv(x: NDArray, y: NDArray, model: RegressorMixin, n_splits: int, random_seed: int, scaler: TransformerMixin = None) -> tuple:
    ''' n_splits-折交叉验证 '''
    assert len(x) == len(y)
    if not scaler: 
        scaler = preprocessing.RobustScaler()
    x = scaler.fit_transform(x)
    kf = KFold(n_splits, shuffle = True, random_state = random_seed)

    r2_buff, mse_buff = [], []
    y_test_buff, y_pred_buff = [], []
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_buff.append(model.score(X_test, y_test))
        mse_buff.append(mean_squared_error(y_test, y_pred))

        y_test_buff += y_test.tolist()
        y_pred_buff += y_pred.tolist()

    print('random seed:', random_seed, ',r2:', r2_score(y_test_buff, y_pred_buff))
    ''' 打印真实值和预测值用于作图 '''
    # for i in range(len(y_test_buff)): print(y_test_buff[i], y_pred_buff[i])   

def parallel_custom_cv_enum_randseed(x: NDArray, 
                                    y: NDArray, 
                                    model: RegressorMixin, 
                                    n_splits: int, 
                                    scaler: TransformerMixin = None, 
                                    n_jobs = -1, 
                                    num_parallel = 25) -> list:
    ''' 
        多线程custom_cv, 测试不同随机数种子下的r^2, 测试数据与模型的稳定性
    '''
    rs_list = [3602, 2594, 1762, 250, 3027, 2978, 3501, 9262, 7850, 9290]
    __res = Parallel(n_jobs = n_jobs)(delayed(custom_cv)(x, y, model, n_splits, rs, scaler) \
                                        for rs in rs_list)

if __name__ == '__main__':
    path = 'Strength_109_corrected_features.xlsx'
    data = pd.read_excel(path, index_col = 0)

    '''
        elem_names: 16种元素名称
        prop_name: 选取的性能名称
        feature_names: 选取的特征名称
        synthetic_val: 人工合成的两个特征, for <strain> only!
        feature_val_part_1: 原始特征
        comp_val: 元素组成
        prop_val: 性能值
        feature_val_part_2: 人工合成的两个特征
        feature_val: 合并后的特征矩阵
    '''
    ''' 
        column 0-15: elements
        column 16: property
    '''
    elem_names = data.columns.values.tolist()[:16]
    prop_name = data.columns.values.tolist()[16]
    ''' selected features based on <ref Zhang Yan, PhD thesis, 2023> '''
    feature_names = ['delta.RAM', 'delta.E', 'delta.NCE', 'delta.MT', 'range.EM', 'range.DVE']
    ''' synthetic feature values based on <ref Zhang Yan, PhD thesis, 2023> '''
    synthetic_val = [
        [0.22, 0.5, 0.37, 0.63, 0.84, 0.22, 0.83, 0.02, 0.87, 0.55, 0.96, 0.73, 0.09, 0.42, 0.48, 0.3],
        [0.6, 0.67, 0.66, 0.61, 0.32, 0.69, 0.85, 0.11, 0.87, 0.65, 0.83, 0.45, 0.82, 0.41, 0.36, 0.56]
    ]

    feature_val_part_1 = data[feature_names].values
    comp_val = data[elem_names].values
    prop_val = data[prop_name].values

    feature_val_part_2 = np.zeros((len(feature_val_part_1), 2))
    for i in range(len(feature_val_part_1)):
        for j in range(len(synthetic_val)):
            ''' 对于生成特征, 仅加权平均值 '''
            feature_val_part_2[i][j] = np.dot(comp_val[i], synthetic_val[j])
    feature_val = np.hstack((feature_val_part_1, feature_val_part_2))

    print('property',prop_name)

    ''' 选取一个随机数种子,打印实验测试性能数值与交叉验证预测结果画图 '''
    # custom_cv(feature_val, prop_val, get_random_forest(), 10, 3673)
    ''' 测试不同随机数种子下的r^2 '''
    parallel_custom_cv_enum_randseed(feature_val, prop_val, get_random_forest(), 10, num_parallel = 8)