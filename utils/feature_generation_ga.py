'''
Copyright (c) 2024 Xian Yuehui (xianyuehui@stu.xjtu.edu.cn), Zhang Yan (
626409903@qq.com) and Xue Dezhen* (xuedezhen@xjtu.edu.cn) of MIL lab from 
Xi'an Jiaotong University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
'''

from __future__ import annotations
import random
import copy
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import r2_score

''' This import can be replaced by a in-file func definition '''
from func_utils import get_random_forest, weight_avg

''' golbal variables '''
'''
    Number of selected atomic features by Genetic Algorithm.

    NOTE: this number should be consistent with the number of 
    features in `feature_name` and the len of self.feature_name 
    in class `Env`.
'''
F_N = 6
'''
    Number of parallel threads in fitness evaluation.

    NOTE: <n_pop // THREAD_N == interger> is recommended.
    But if n_pop is very large, then THREAD_N should be the
    usable threads of your CPU.
'''
THREAD_N = 10
# number of parallel jobs in sklearn cross validation, NOT USED NOW
JOBS_N = 2
# number of splits for cross validation
SPLITS_N = 10
# number of elements, how many components in a comosition
E_N = 16
'''
    Number of generated features

    NOTE: 2 is borrowed from Ref_2023_PhD_Thesis_ZhangYan
'''
G_F_N = 2
'''
    Number of 2-bit (0 or 1) digits for one generated features.

    NOTE: [0.0, 1.0] with a step of 0.1 from Ref_2023_PhD_Thesis_ZhangYan
    is recommended. But here we use a 8-bit interger to improve the
    original implementation. For example, in the original implementation,
    the mutation operation is to randomly choose another float in [0.0, 1.0]
    with a step of 0.1. But in our implementation, the mutation operation is
    to randomly flip one bit in the 8-bit interger. This can make the mutation
    operation do not lose most original information. I.e., if the flip bit is
    on the left, then the new value is just slightly twisted from the 
    original value.

    WARNING: careflul when changing this value, it should be consistent with
    the data type in generated features array.
'''
D_N = 8

class Env:
    '''
        Environment for feature generation.
    '''
    def __init__(self) -> None:
        '''
            Environment initialization.

            NOTE: FOR USERS, The first 5 lines can be modified according to specific purpose.
        '''
        initial_dataset_path = 'Strain_final_118_checked_by_Xue_20240110.xlsx'
        data = pd.read_excel(initial_dataset_path, index_col = 0)
        self.elem_names = data.columns.values.tolist()[:16]
        self.prop_name = data.columns.values.tolist()[16]

        '''
            Selected strain features by Genetic Algorithm.

            NOTE: this list should be USER DEFINED and carefully examined
            before running the GA.
        '''
        self.feature_name = data.columns.values[[22, 35, 63, 112, 188, 211]].tolist()

        self.comp_val = data[self.elem_names].values
        self.prop_val = data[self.prop_name].values
        self.feature_val = data[self.feature_name].values
        self.feature_num = self.feature_val.shape[1]
        self.model = get_random_forest()
        self.scaler = preprocessing.RobustScaler()
    
    def __skl_cross_validation(self, x, y) -> float:
        '''
            10-fold cross validation using sklearn.

            NOTE: deprecated. shift to custom cross validation.
        '''
        scores = cross_val_score(self.model, x, y, cv = SPLITS_N, scoring = 'r2', n_jobs = JOBS_N)
        return scores.mean()

    def __custom_cross_validation(self, x, y) -> float:
        ''' 
            Custom 10-fold cross validation.
        '''
        assert len(x) == len(y)

        model = clone(self.model)
        scaler = preprocessing.RobustScaler()
        x = scaler.fit_transform(x)
        kf = KFold(SPLITS_N, shuffle = True)

        y_test_buff, y_pred_buff = [], []
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_test_buff += y_test.tolist()
            y_pred_buff += y_pred.tolist()

        return r2_score(y_test_buff, y_pred_buff)

    def calculate_fitness(self, ind: Individual):
        '''' calculate fitness of an individual '''
        # return random.random()  # for smoke test
        elem_f_arr = ind.elem_f_arr

        ''' calculate the generated features using the elem_f_arr and the composition '''
        gen_f_arr = np.dot(self.comp_val, elem_f_arr.T)
        ''' concat the original features and the generated features '''
        x = np.concatenate((self.feature_val, gen_f_arr), axis = 1)

        x = self.scaler.fit_transform(x)
        fitness = self.__custom_cross_validation(x, self.prop_val)
        
        return fitness

class Individual:
    '''
        Individual in the population of Genetic Algorithm.
    '''
    def __init__(self, elem_f_arr: np.ndarray):
        self.elem_f_arr = elem_f_arr
        self.fitness = None

def init_individual() -> Individual:
    """
        Initializes an individual with randomly generated features.

        Returns:
            Individual: The initialized individual.
    """
    elem_f_arr = np.random.randint(-128, 127, size=(G_F_N, E_N), dtype=np.int8)
    return Individual(elem_f_arr)

def softmax(x: List[float]) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

def select(population: List[Individual], k: int) -> List[Individual]:
    """
        Select the individuals from a population.

        轮盘赌选择(Roulette Wheel Selection)
        在这种方法中，每个个体被选中的概率与其适应度函数值的大小正相关。

        :param population: The population.
        :param k: The number of individuals to select.
        :return: The selected individuals.
    """
    fitness_list = [individual.fitness for individual in population]
    ''' fitness defined as r2 can be negative, need to normalize it to [0, 1] '''
    probabilities = softmax(fitness_list)
    sel_idx = np.random.choice(len(population), size = k, replace = False, p = probabilities)
    return [population[i] for i in sel_idx]

def mutate(parent: Individual, mutation_prob: float) -> Individual:
    """
        Mutates the given parent Individual by flipping bits in 
        the elem_f_arr based on the mutation probability.

        Args:
            parent (Individual): The parent Individual to be mutated.
            mutation_prob (float): The probability of mutation for each bit.

        Returns:
            Individual: The mutated Individual.
    """
    elem_f_arr = copy.deepcopy(parent.elem_f_arr)
    for i in range(G_F_N):
        for j in range(E_N):
            if random.random() < mutation_prob:
                '''
                    Assume `x` is your int8 variable and `n` is the bit you want to flip
                '''
                n = random.randint(0, D_N - 1)
                elem_f_arr[i][j] ^= 1 << n
    
    return Individual(elem_f_arr)

def cross_over(parent_1: Individual, parent_2: Individual, crx_pb: float = 0.25) -> Tuple[Individual, Individual]:
    '''
        Perform crossover and mutation on two parent individuals.

        Args:
            parent_1 (Individual): The first parent individual.
            parent_2 (Individual): The second parent individual.
            crx_pb (float, optional): The crossover probability. Defaults to 0.25.

        Returns:
            Tuple[Individual, Individual]:  A tuple containing two child individuals 
            resulting from crossover and mutation.
    '''
    elem_f_arr_1 = copy.deepcopy(parent_1.elem_f_arr)
    elem_f_arr_2 = copy.deepcopy(parent_2.elem_f_arr)
    for i in range(G_F_N):
        for j in range(E_N):
            if random.random() < crx_pb:
                elem_f_arr_1[i][j], elem_f_arr_2[i][j] = elem_f_arr_2[i][j], elem_f_arr_1[i][j]
    return Individual(elem_f_arr_1), Individual(elem_f_arr_2)

def elitism_replacement(population: List[Individual], offspring: List[Individual]):
    """ Perform Elitism Replacement """
    combined = population + offspring
    combined.sort(key = lambda ind: ind.fitness, reverse = True)
    return combined[:len(population)]

class FeatureGenrationGA:
    """
        FeatureGenrationGA
        This class uses Genetic Algorithm to syhtetically generate features for the given data.

        遗传算法通常包含以下几个阶段:
        初始化(Initialization):创建一个初始种群。这个种群通常是随机生成的。
        评估(Evaluation):评估种群中每个个体的适应度。
        选择(Selection):<根据每个个体的适应度来选择用于交叉的个体。适应度高的个体有更高的机会被选中。>
        交叉(Crossover):从已经选择的个体中创建新的个体。这个过程模拟了生物的交配过程。
        突变(Mutation):对新生成的个体进行随机的小修改，这个过程模拟了生物的突变过程。
        替换(Replacement):用新生成的个体替换掉种群中的一部分或全部个体。
        终止(Termination).
    """

    def __init__(self, ff_obj: Env, verbose: int = 0):
        """
        Parameters
        -----------
        ff_obj: {object}, environment for feature generation
        verbose: 0 or 1
        """
        self.verbose = verbose
        # self.final_fitness = []
        self.dominants_buffer = {}
        self.best_ind = None
        if ff_obj == None:
            raise ValueError("Please provide a valid environment.")
        else:
            self.env = ff_obj

        if self.verbose == 1:
            print(
                "Will generate {} features arrays".format(G_F_N)
            )
            print("Shape of train_x: {} and target: {}".format(self.env.feature_val.shape, self.env.prop_val.shape))

    def par_eval(self, pop: List[Individual]) -> List[float]:
        ''' parallel evaluation of fitness '''
        fitnesses = joblib.Parallel(n_jobs=THREAD_N)(joblib.delayed(self.env.calculate_fitness)(x) for x in pop)
        return fitnesses

    def generate(self, n_pop, cxpb=0.5, mutxpb=0.2, ngen=5):

        """
        Generate evolved population
        Parameters
        -----------
            n_pop : {int}
                    population size
            cxpb  : {float}
                    crossover probablity
            mutxpb: {float}
                    mutation probablity
            n_gen : {int}
                    number of generations
            set_toolbox : {boolean}
                          If True then you have to create custom toolbox before calling
                          method. If False use default toolbox.
        Returns
        --------
            Fittest population
        """

        if self.verbose == 1:
            print(
                "Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(
                    n_pop, cxpb, mutxpb, ngen
                )
            )

        pop = [init_individual() for _ in range(n_pop)]

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = self.par_eval(pop)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit

        for g in range(ngen):
            self.dominants_buffer[g] = max(pop, key = lambda ind: ind.fitness)

            print(" GENERATION {} ".format(g + 1).center(25, '-'))
            print("Best fitness: {}".format(self.dominants_buffer[g].fitness))
            # self.review_pop(pop)

            selected_pop = select(pop, len(pop) // 2)
            new_individuals = []
            # Apply crossover and mutation on the offspring
            for ind_1, ind_2 in zip(selected_pop[::2], selected_pop[1::2]):     # TODO: check cross_over order
                new_individuals += list(cross_over(ind_1, ind_2, cxpb))

            for ind in selected_pop:
                new_individuals.append(mutate(ind, mutxpb))

            # Evaluate the new individuals
            fitnesses = self.par_eval(new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness = fit
            print("Evaluated %i new individuals" % len(new_individuals))

            # replacement
            pop = elitism_replacement(pop, new_individuals)                         

        print("-- Only the fittest survives --")

        self.best_ind = max(pop, key = lambda ind: ind.fitness)
        print(
            "Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness)
        )
        # self.get_final_scores(pop, fits)

        return pop

    def save_dominants_buffer(self, file_name: str):
        """
            Save fitness in each generation in a file
        """
        joblib.dump(self.dominants_buffer, file_name)

    def review_pop(self, pop: List[Individual]):
        """
            Review population by statistics
        """
        fits = [ind.fitness for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        if self.verbose == 1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

if __name__ == "__main__":
    ''' in-file test '''
    env = Env()
    ga = FeatureGenrationGA(env, verbose = 1)
    ga.generate(n_pop = 200, cxpb = 0.8, mutxpb = 0.1, ngen = 50)
    ga.save_dominants_buffer('dominants_buffer.pkl')