'''
    Al Hf Nb Zr Ti Ta V
'''
import copy
import itertools
import joblib
import numpy as np
import os
from typing import List
from joblib.parallel import Parallel, delayed
import pandas

from surrogate import CompressionStengthSurrogate, CompressionStrainSurrogate, Surrogate
from deprecated import deprecated

ELEMS = ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V']
COMP_LOW_LIMITS = np.array([0., 0., 0., 0., 0., 0., 0.])
COMP_HIGH_LIMITS = np.array([14., 34., 34., 34., 34., 34., 34.])
COMP_STEP = 2.
MIN_ELEM_NUM = 3    # no need
COMP_ROUNDING_DIGITS = 1

counter = 0

def recursive_enumeration(current_comp: List[float], \
                          unassigned_elem_idxs: List[int], \
                          possible_compositons: List[float]):
    '''
        Efficiently enumerate all possible compositions using DFS algorithm.

        current_comp:           List[float], current composition
        unassigned_elem_idxs:   List[int], unassigned element indexes
        possible_compositons:   List[float], all possible compositions
    '''
    if len(unassigned_elem_idxs) == 1:
        tmp_comp = current_comp + [round(100. - sum(current_comp), COMP_ROUNDING_DIGITS)]
        ''' NO NEED '''
        # if sum([1 if round(c, COMP_ROUNDING_DIGITS) != 0. else 0 for c in tmp_comp]) >= MIN_ELEM_NUM:
        #     possible_compositons.append(tmp_comp)
        possible_compositons.append(tmp_comp)
        if len(possible_compositons) % int(1e6) == 0:
            global counter
            joblib.dump(possible_compositons, f'all_possible_compositons_{COMP_STEP}_{counter}.pkl')
            possible_compositons.clear()
            counter += 1
    else:
        idx = unassigned_elem_idxs[0]
        unassigned_comp = 100. - sum(current_comp)
        tmp_c_low = max(COMP_LOW_LIMITS[idx], unassigned_comp - sum(COMP_HIGH_LIMITS[unassigned_elem_idxs[1:]]))
        tmp_c_high = min(COMP_HIGH_LIMITS[idx], unassigned_comp - sum(COMP_LOW_LIMITS[unassigned_elem_idxs[1:]]))
        for c in [round(tmp_c_low + n * COMP_STEP, COMP_ROUNDING_DIGITS) \
                  for n in range(round((tmp_c_high - tmp_c_low) / COMP_STEP + 1))]:
            if c % 2:
                raise Exception('c % 2 != 0')
            current_comp.append(c)
            recursive_enumeration(current_comp, unassigned_elem_idxs[1:], possible_compositons)
            current_comp.pop()

@deprecated(version='0.1', reason='deprecated parallel_acq_func')
def paral_acq_func(comp_f_path: str, 
                   surr: Surrogate, 
                   best_so_far: float, 
                   epsilon: float = 0.0, 
                   ucb_kappa: float = 1.96
                   ) -> List[float]:
    '''
        Base function for parallel acq_func calculation.

        comp_f_path:    str, path of composition file
        surr:           Surrogate, surrogate model
        best_so_far:    float, ref value in EI acquisition function
    '''
    _surr = copy.deepcopy(surr)
    _comps = joblib.load(comp_f_path)
    ucbs, eis = _surr.acq_func(_comps, best_so_far, epsilon, ucb_kappa)
    ucb_max_idx = np.argmax(ucbs)
    ei_max_idx = np.argmax(eis)
    return (_comps[ucb_max_idx], ucbs[ucb_max_idx]), (_comps[ei_max_idx], eis[ei_max_idx])

if __name__ == '__main__':
    gen_all_comps = False
    strain_best_so_far = 55.
    strength_best_so_far = 3882
    base_dir = '.'
    
    if gen_all_comps:
        ''' enumerate and dump all possible compositions '''
        idxs = list(range(len(ELEMS)))
        possible_compositons = []
        recursive_enumeration([], idxs, possible_compositons)
        joblib.dump(possible_compositons, 
                    os.path.join(base_dir, f'all_possible_compositons_{COMP_STEP}_{counter}.pkl'))
    else:
        ''' enumerate utility value in parallel '''
        # num_threads = 12
        # comp_path_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir) \
        #                     if f.startswith(f'all_possible_compositons_{COMP_STEP}')]
        
        # _surr = CompressionStrainSurrogate()
        # parl_res = Parallel(n_jobs = num_threads)(delayed(paral_acq_func)(comp_path, _surr, best_so_far) for comp_path in comp_path_list)
        # ucb_res, ei_res = zip(*parl_res)
        # ucb_max_idx = np.argmax([r[1] for r in ucb_res])
        # ei_max_idx = np.argmax([r[1] for r in ei_res])

        # proposition = {
        #     'ucb_res': ucb_res,
        #     'ei_res': ei_res,
        #     'ucb_max': ucb_res[ucb_max_idx][1],
        #     'ei_max': ei_res[ei_max_idx][1],
        #     'ucb_comp': ucb_res[ucb_max_idx][0],
        #     'ei_comp': ei_res[ei_max_idx][0]
        # }

        # for k, v in proposition.items(): print(k, v)
        # joblib.dump(proposition, f'res_{COMP_STEP}.pkl')

        strain_surr = CompressionStrainSurrogate()
        strength_surr = CompressionStengthSurrogate()

        comp_buffer = []
        strain_mean_buffer = []
        strain_ucb_buffer = []
        strain_ei_buffer = []

        strength_mean_buffer = []
        strength_ucb_buffer = []
        strength_ei_buffer = []

        component_dict = dict(zip(range(1, len(ELEMS) + 1), [[] for _ in range(len(ELEMS))]))

        comp_path_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir) \
                            if f.startswith(f'all_possible_compositons_{COMP_STEP}')]
        
        for comp_path in comp_path_list:
            comps = joblib.load(comp_path)
            comp_buffer += comps

            strain_mean, strain_ucb, strain_ei = strain_surr.acq_func(comps, strain_best_so_far)
            strain_mean_buffer += list(strain_mean)
            strain_ucb_buffer += list(strain_ucb)
            strain_ei_buffer += list(strain_ei)

            strength_mean, strength_ucb, strength_ei = strength_surr.acq_func(comps, strength_best_so_far)
            strength_mean_buffer += list(strength_mean)
            strength_ucb_buffer += list(strength_ucb)
            strength_ei_buffer += list(strength_ei)

        # joblib.dump((comp_buffer, 
        #             strain_mean_buffer, 
        #             strain_ucb_buffer, 
        #             strain_ei_buffer, 
        #             strength_mean_buffer, 
        #             strength_ucb_buffer, 
        #             strength_ei_buffer), 'all_res.pkl'), exit()

        # comp_buffer, \
        #     strain_mean_buffer, \
        #     strain_ucb_buffer, \
        #     strain_ei_buffer, \
        #     strength_mean_buffer, \
        #     strength_ucb_buffer, \
        #     strength_ei_buffer = joblib.load('all_res.pkl')

        # for c, m1, u1, e1, m2, u2, e2 in zip(comp_buffer, 
        #                                      strain_mean_buffer, 
        #                                      strain_ucb_buffer, 
        #                                      strain_ei_buffer, 
        #                                      strength_mean_buffer, 
        #                                      strength_ucb_buffer, 
        #                                      strength_ei_buffer):
        #     if m1 < 45:
        #         print(c, m1, u1, e1, m2, u2, e2)
        # #     # c中非零数
        # #     component_idx = np.array(c, dtype = np.bool_).sum()
        # #     component_dict[component_idx].append((c, m1, u1, e1, m2, u2, e2))
        
        # ''' 防化院的要求: 10 < strain < 20, 1500 < strength < 2500, 强度越大越好 '''
        for c, m1, u1, e1, m2, u2, e2 in zip(comp_buffer, 
                                             strain_mean_buffer, 
                                             strain_ucb_buffer, 
                                             strain_ei_buffer, 
                                             strength_mean_buffer, 
                                             strength_ucb_buffer, 
                                             strength_ei_buffer):
            # c中非零数
            component_idx = np.array(c, dtype = np.bool_).sum()
            component_dict[component_idx].append([c, m1, u1, e1, m2, u2, e2])
        
        buffer = []
        for k, v in component_dict.items():
            if len(v):
                selected_items = [item for item in v if 15 < item[1] < 20 and 2000 < item[4] < 2500]
                selected_items.sort(key = lambda x: x[4], reverse = True)    # 按照强度降序排列
                selected_items = [item[0] + [k] + item[1:] for item in selected_items]
                buffer += selected_items
        
        pandas.DataFrame(
            buffer,
            columns = ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V', 'component number', 'strain_mean', 'strain_ucb', 'strain_ei', 'strength_mean', 'strength_ucb', 'strength_ei']
        ).to_excel(f'predictions.xlsx')

        ''' 实验迭代 '''
        buffer = []
        acq_name = list(map(lambda x: x + '_max', ['strain_mean', 'strain_ucb', 'strain_ei', 'strength_mean', 'strength_ucb', 'strength_ei']))
        for k, v in component_dict.items():
            if len(v):
                tmp_buffer = []
                for i in range(1, 7):
                    selected_item = max(v, key = lambda x: x[i])
                    tmp_buffer.append(selected_item)
                buffer += [item[0] + [k] + item[1:] + [an] for item, an in zip(tmp_buffer, acq_name)]
        pandas.DataFrame(
            buffer,
            columns = ['Al', 'Hf', 'Nb', 'Zr', 'Ti', 'Ta', 'V', 'component number', 'strain_mean', 'strain_ucb', 'strain_ei', 'strength_mean', 'strength_ucb', 'strength_ei', 'utility function']
        ).to_excel(f'proposition.xlsx')
            