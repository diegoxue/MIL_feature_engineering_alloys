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
from func_utils import get_random_forest

''' golbal variables '''
# feature number
F_N = 6
# thread number
THREAD_N = 10
# number of parallel jobs in cross validation
JOBS_N = 2
# number of splits for cross validation
SPLITS_N = 10

class Env:
    '''
        Environment for feature selection of strain data.
    '''
    def __init__(self) -> None:
        '''
            Environment initialization.
            The first 5 lines can be modified according to specific purpose.
        '''
        initial_dataset_path = 'Strain_final_118_checked_by_Xue_20240110.xlsx'
        data = pd.read_excel(initial_dataset_path, index_col = 0)
        _elem_names = data.columns.values.tolist()[:16]
        self.prop_name = data.columns.values.tolist()[16]
        self.feature_name = data.columns.values.tolist()[17:]


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
        # return random.random()  # for debug

        feature_idx_list = np.asarray(ind.f_idx_list, dtype = np.int32)

        assert len(feature_idx_list) == F_N
        assert len(set(feature_idx_list)) == F_N
        assert feature_idx_list.min() >= 0 and feature_idx_list.max() < self.feature_num
        
        x = self.feature_val[:, feature_idx_list]
        x = self.scaler.fit_transform(x)
        fitness = self.__custom_cross_validation(x, self.prop_val)
        
        return fitness
    
    @property
    def total_f_N(self) -> int:
        return self.feature_num

class Individual:
    def __init__(self, f_idx_list: List[int]):
        ''' feature index list, type: List[int] '''
        self.f_idx_list = np.array(f_idx_list)
        self.fitness = None

def init_individual(total_f_num: int, f_num: int) -> Individual:
    """
        Generate an individual.

        :param total_f_num: The total number of features.
        :param f_num: The number of features to select.
        :return: An individual.
    """
    f_idx_list = random.sample(range(total_f_num), f_num)
    return Individual(f_idx_list)

def softmax(x: List[float]) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

def select(population: List[Individual], k: int) -> List[Individual]:
    """
        Select the individuals from a population.

        轮盘赌选择(Roulette Wheel Selection)
        在这种方法中，每个个体被选中的概率与其适应度函数值的大小成正比。
        具体来说，每个个体的选择概率等于其适应度函数值除以所有个体适应度函数值的总和。
        这种方法更倾向于选择适应度高的个体，但也给适应度低的个体留下了一定的选择机会。

        :param population: The population.
        :param k: The number of individuals to select.
        :return: The selected individuals.
    """
    fitness_sum = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / fitness_sum for individual in population]
    ''' fitness defined as r2 can be negative, need to normalize it to [0, 1] '''
    probabilities = softmax(probabilities)
    return random.choices(population, weights = probabilities, k = k)

def mutate(parent: Individual, mutation_prob: float, total_f_num: int) -> Individual:
    """
        Mutate an individual by replacing one of its attribute with a random integer value.

        :param individual: The individual to be mutated.
        :return: A new mutated individual.
    """
    f_idx_list = copy.deepcopy(parent.f_idx_list)
    mutate_pos = [i for i in range(len(f_idx_list)) if random.random() < mutation_prob]
    mutate_val = random.sample(
        list(set(range(total_f_num)) - set(f_idx_list)), 
        len(mutate_pos)
    )
    f_idx_list[mutate_pos] = mutate_val
    return Individual(f_idx_list)

def cross_over(parent_1: Individual, parent_2: Individual, crx_pb: float = 0.25) -> Tuple[Individual, Individual]:
    """
        Cross over two individuals.

        :param individual1: The first individual.
        :param individual2: The second individual.
        :return: Two new crossed individuals.
    """
    f_idx_list_1 = copy.deepcopy(parent_1.f_idx_list)
    f_idx_list_2 = copy.deepcopy(parent_2.f_idx_list)
    only_in_1 = list(set(f_idx_list_1) - set(f_idx_list_2))
    only_in_2 = list(set(f_idx_list_2) - set(f_idx_list_1))
    ''' shuffle '''
    random.shuffle(only_in_1)
    random.shuffle(only_in_2)
    ''' cross over '''
    new_f_idx_list_1 = list(set(f_idx_list_1) & set(f_idx_list_2)) + only_in_1
    new_f_idx_list_2 = list(set(f_idx_list_1) & set(f_idx_list_2)) + only_in_2
    for i in range(len(only_in_1)):
        if random.random() < crx_pb:
            new_f_idx_list_1[i], new_f_idx_list_2[i] = new_f_idx_list_2[i], new_f_idx_list_1[i]
    return Individual(new_f_idx_list_1), Individual(new_f_idx_list_2)

def elitism_replacement(population: List[Individual], offspring: List[Individual]):
    """ Perform Elitism Replacement """
    combined = population + offspring
    combined.sort(key = lambda ind: ind.fitness, reverse = True)
    return combined[:len(population)]

class FeatureSelectionGA:
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for the given data.

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
        ff_obj: {object}, environment for feature selection
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
                "Will select best features among {} features.".format(self.env.feature_val.shape[1])
            )
            print("Shape od train_x: {} and target: {}".format(self.env.feature_val.shape, self.env.prop_val.shape))

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

        pop = [init_individual(self.env.total_f_N, F_N) for _ in range(n_pop)]

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
                new_individuals.append(mutate(ind, mutxpb, self.env.total_f_N))

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
    env = Env()
    ga = FeatureSelectionGA(env, verbose = 1)
    ga.generate(n_pop = 200, cxpb = 0.8, mutxpb = 0.1, ngen = 50)
    ga.save_dominants_buffer('dominants_buffer.pkl')