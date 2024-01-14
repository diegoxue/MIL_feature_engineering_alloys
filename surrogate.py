import random
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import datasets, ensemble, preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, Matern,
                                              RationalQuadratic, WhiteKernel)
from numpy.typing import NDArray
from func_utils import F_func, avg_func, delta_func, min_func, minc_func, range_func, rangec_func, get_random_forest



class Surrogate:
    ''' Abstract class for surrogate model. Do not instantiate this class. '''
    def __init__(self) -> None:
        raise Exception('Do not instantiate this class.')
        
    def acq_func(self, compositions: List[List[float]], 
                 best_so_far: float, 
                 epsilon: float = 0.0, 
                 ucb_kappa: float = 1.96) -> Tuple[List[float], List[float]]:
        pass

    def cal_atomic_features(self, compositions: List[List[float]]) -> NDArray:
        pass

class CompressionStrainSurrogate(Surrogate):
    def __init__(self, initial_dataset_path: str = 
            'Strain_final_118_checked_by_Xue_20240110.xlsx'):
        ''' load training data '''
        data = pd.read_excel(initial_dataset_path, index_col = 0)
        elem_names = data.columns.values.tolist()[:16]
        prop_name = data.columns.values.tolist()[16]
        feature_names = ['minc.C', 'rangec.VEC', 'delta.SIE', 'range.EM', 'min.D', 'F']
        synthetic_val = [
            [0.88, 0.73, 0.41, 0.62, 0.83, 0.4, 0.93, 0.82, 0.94, 0.19, 0.44, 0.02, 0.39, 0.74, 0.97, 0.98],
            [0.4, 0.5, 0.89, 0.37, 0.25, 0.01, 0.88, 0.17, 0.58, 0.38, 0.5, 0.78, 0.02, 0.58, 0.6, 0.46]
        ]
        
        feature_val_part_1 = data[feature_names].values
        comp_val = data[elem_names].values
        prop_val = data[prop_name].values

        feature_val_part_2 = np.zeros((len(feature_val_part_1), 2))
        for i in range(len(feature_val_part_1)):
            for j in range(len(synthetic_val)):
                feature_val_part_2[i][j] = np.dot(comp_val[i], synthetic_val[j])
        feature_val = np.hstack((feature_val_part_1, feature_val_part_2))
        self.feature_len = len(feature_val[0])

        ''' fit the model '''
        '''
            Model: Random Frorest Regressor
            X:  Scaled ['minc.C', 'rangec.VEC', 'delta.SIE', 'range.EM', 'min.D', 'F', 'synth_1', 'synth_2']
                ->
            Y:  Strain prediction.
        '''
        self.model = get_random_forest()
        self.scaler = preprocessing.RobustScaler()
        x = self.scaler.fit_transform(feature_val)
        self.model.fit(x, prop_val)

        ''' features parameters '''
        _elem_feature = pd.read_csv('ELEMENTS16-NEW.csv', index_col = 0)
        ''' NOTE: F feature is separately calculated '''
        sel_f_names = ['C', 'VEC', 'SIE', 'EM', 'D']
        self.elem_feature = np.concatenate((_elem_feature.loc[sel_f_names].values, synthetic_val), axis = 0)
        self.sel_f = [minc_func, rangec_func, delta_func, range_func, min_func, avg_func, avg_func]   # TODO
        # e_list, r_list, g_list
        self.e_list = _elem_feature.loc['E'].values
        self.r_list = _elem_feature.loc['R'].values
        self.g_list = _elem_feature.loc['G'].values
        assert len(self.elem_feature) == len(self.sel_f)

    def cal_atomic_features(self, compositions: List[List[float]]) -> NDArray:
        ''' calculate atomic features of comp ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V'] '''
        compositions = np.array(compositions)
        assert compositions.shape[-1] == 7 or compositions.shape[-1] == 16
        ''' 
            ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V']
            ->
            ['Al', 'Si', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']
            
            Al	Si	Ti	V	Cr	Mn	Fe	Co	Ni	Cu	Zr	Nb	Mo	Hf	Ta	W
            0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15

            new_comp[[0, 13, 11, 10, 2, 14, 3]] = comp
        '''
        new_comp = np.zeros((len(compositions), 16))
        new_comp[:, [0, 13, 11, 10, 2, 14, 3]] = compositions

        ''' ['minc.C', 'rangec.VEC', 'delta.SIE', 'range.EM', 'min.D', 'F', 'synth_1', 'synth_2'] '''
        ''' NOTE: 最后单独计算 F feature '''
        f_array = np.zeros((len(compositions), self.feature_len))
        for i in range(len(self.sel_f)):
            for j in range(len(f_array)):
                f_array[j][i] = self.sel_f[i](new_comp[j], self.elem_feature[i])
        f_array[:, -2:] = f_array[:, -3: -1]

        for i in range(len(f_array)):
            f_array[i][-3] = F_func(new_comp[i], self.e_list, self.r_list, self.g_list)
        
        return f_array
    
    def acq_func(self, compositions: List[List[float]], 
                 best_so_far: float, 
                 epsilon: float = 0.0, 
                 ucb_kappa: float = 1.96) -> Tuple[List[float], List[float]]:
        '''
            compositions:   List[List[float]]
            best_so_far:    float, ref value in EI acquisition function
            epsilon:        float, exploration parameter in EI acquisition function
            ucb_kappa:      float, exploration parameter in UCB acquisition function
        '''
        atomic_features = self.cal_atomic_features(compositions)
        scaled_x = self.scaler.transform(atomic_features)

        ''' prediction '''
        mean = self.model.predict(scaled_x)
        preds = np.array([estimator.predict(scaled_x) for estimator in self.model.estimators_])
        std = np.std(preds, axis=0)

        ''' calculate ucb and ei '''
        ucb = mean + ucb_kappa * std
        x_prime = (mean - best_so_far) / std - epsilon
        ei = (mean - best_so_far - epsilon * std) * norm.cdf(x_prime) + std * norm.pdf(x_prime)

        return mean, ucb, ei

class CompressionStengthSurrogate(Surrogate):
    def __init__(self, initial_dataset_path: str = 
            'Strength_109_corrected_features.xlsx'):
        ''' load training data '''
        data = pd.read_excel(initial_dataset_path, index_col = 0)
        elem_names = data.columns.values.tolist()[:16]
        prop_name = data.columns.values.tolist()[16]
        feature_names = ['delta.RAM', 'delta.E', 'delta.NCE', 'delta.MT', 'range.EM', 'range.DVE']
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
                feature_val_part_2[i][j] = np.dot(comp_val[i], synthetic_val[j])
        feature_val = np.hstack((feature_val_part_1, feature_val_part_2))
        self.feature_len = len(feature_val[0])

        ''' fit the model '''
        '''
            Model: Random Frorest Regressor
            X:  Scaled ['delta.RAM', 'delta.E', 'delta.NCE', 'delta.MT', 'range.EM', 'range.DVE', 'synth_1', 'synth_2']
                ->
            Y:  strength prediction.
        '''
        self.model = get_random_forest()
        self.scaler = preprocessing.RobustScaler()
        x = self.scaler.fit_transform(feature_val)
        self.model.fit(x, prop_val)

        ''' features parameters '''
        _elem_feature = pd.read_csv('ELEMENTS16-NEW.csv', index_col = 0)
        ''' NOTE: F feature is separately calculated '''
        sel_f_names = ['RAM', 'E', 'NCE', 'MT', 'EM', 'DVE']
        self.elem_feature = np.concatenate((_elem_feature.loc[sel_f_names].values, synthetic_val), axis = 0)
        self.sel_f = [delta_func, delta_func, delta_func, delta_func, range_func, range_func, avg_func, avg_func]   # TODO
        assert len(self.elem_feature) == len(self.sel_f)

    def cal_atomic_features(self, compositions: List[List[float]]) -> NDArray:
        ''' calculate atomic features of comp ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V'] '''
        compositions = np.array(compositions)
        assert compositions.shape[-1] == 7 or compositions.shape[-1] == 16
        ''' 
            ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V']
            ->
            ['Al', 'Si', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']
            
            Al	Si	Ti	V	Cr	Mn	Fe	Co	Ni	Cu	Zr	Nb	Mo	Hf	Ta	W
            0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15

            new_comp[[0, 13, 11, 10, 2, 14, 3]] = comp
        '''
        new_comp = np.zeros((len(compositions), 16))
        new_comp[:, [0, 13, 11, 10, 2, 14, 3]] = compositions

        ''' ['delta.RAM', 'delta.E', 'delta.NCE', 'delta.MT', 'range.EM', 'range.DVE', 'synth_1', 'synth_2'] '''
        f_array = np.zeros((len(compositions), self.feature_len))
        for i in range(len(self.sel_f)):
            for j in range(len(f_array)):
                f_array[j][i] = self.sel_f[i](new_comp[j], self.elem_feature[i])
        
        return f_array
    
    def acq_func(self, compositions: List[List[float]], 
                 best_so_far: float, 
                 epsilon: float = 0.0, 
                 ucb_kappa: float = 1.96) -> Tuple[List[float], List[float]]:
        '''
            compositions:   List[List[float]]
            best_so_far:    float, ref value in EI acquisition function
            epsilon:        float, exploration parameter in EI acquisition function
            ucb_kappa:      float, exploration parameter in UCB acquisition function
        '''
        atomic_features = self.cal_atomic_features(compositions)
        scaled_x = self.scaler.transform(atomic_features)

        ''' prediction '''
        mean = self.model.predict(scaled_x)
        preds = np.array([estimator.predict(scaled_x) for estimator in self.model.estimators_])
        std = np.std(preds, axis=0)

        ''' calculate ucb and ei '''
        ucb = mean + ucb_kappa * std
        x_prime = (mean - best_so_far) / std - epsilon
        ei = (mean - best_so_far - epsilon * std) * norm.cdf(x_prime) + std * norm.pdf(x_prime)

        return mean, ucb, ei

if __name__ == '__main__':
    compression_model = CompressionStrainSurrogate()
    best_so_far = 55.
    comps = joblib.load('test_comps.pkl')
    comps = comps[:10]
    # print(compression_model.acq_func(comps, best_so_far))