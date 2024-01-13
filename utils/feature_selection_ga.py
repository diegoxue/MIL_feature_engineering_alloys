import random
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from func_utils import get_random_forest

import joblib

# feature number
F_N = 6
# number of parallel jobs
JOBS_N = 5
# number of splits for cross validation
SPLITS_N = 10

class Env:
    '''
        Environment for feature selection of strain data.
    '''
    def __init__(self) -> None:
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
    
    def _cross_validation(self, x, y):
        scores = cross_val_score(self.model, x, y, cv = SPLITS_N, scoring = 'r2', n_jobs = JOBS_N)
        return scores.mean()

    def calculate_fitness(self, feature_idx_list: List[int]):
        feature_idx_list = np.asarray(feature_idx_list, dtype = np.int32)

        assert len(feature_idx_list) == F_N
        assert len(set(feature_idx_list)) == F_N
        assert feature_idx_list.min() >= 0 and feature_idx_list.max() < self.feature_num
        
        x = self.feature_val[:, feature_idx_list]
        x = self.scaler.fit_transform(x)
        fitness = self._cross_validation(x, self.prop_val)
        
        return fitness
    
    @property
    def total_f_N(self) -> int:
        return self.feature_num

class Individual:
    def __init__(self, f_idx_list: List[int]):
        self.f_idx_list = f_idx_list
        self.fitness = None

def init_individual(total_f_num: int, f_num: int):
    """
        Generate an individual.

        :param total_f_num: The total number of features.
        :param f_num: The number of features to select.
        :return: An individual.
    """
    f_idx_list = random.sample(range(total_f_num), f_num)
    return Individual(f_idx_list)

def mutate(individual: Individual, total_f_num: int):
    """
        Mutate an individual by replacing one of its attribute with a random integer value, in place.

        :param individual: The individual to be mutated.
        :return: A mutated individual.
    """
    f_idx_list = individual.f_idx_list
    mutate_pos = random.randint(0, len(f_idx_list) - 1)
    mutate_val = random.choice(set(range(total_f_num)) - set(f_idx_list))
    f_idx_list[mutate_pos] = mutate_val
    return individual

def cross_over(individual_1: Individual, individual_2: Individual, crx_pb: float = 0.25):
    """
        Cross over two individuals.

        :param individual1: The first individual.
        :param individual2: The second individual.
        :return: Two crossed individuals.
    """
    f_idx_list_1 = individual_1.f_idx_list
    f_idx_list_2 = individual_2.f_idx_list
    cross_pos = random.randint(0, len(f_idx_list_1) - 1)
    if random.random() < crx_pb and f_idx_list_1[cross_pos] != f_idx_list_2[cross_pos]:
        f_idx_list_1[cross_pos], f_idx_list_2[cross_pos] = f_idx_list_2[cross_pos], f_idx_list_1[cross_pos]
    return individual_1, individual_2

def select(population: List[Individual], k: int) -> List[Individual]:
    """
        Select the best individuals from a population.

        :param population: The population.
        :param k: The number of individuals to select.
        :return: The selected individuals.
    """
    _POOL_SIZE = 3
    assert _POOL_SIZE <= len(population)
    selected_pop = []
    for _ in range(k):
        selected_pop.append(
            max(random.sample(population, _POOL_SIZE), key = lambda individual: individual.fitness)
        )
    return selected_pop

class FeatureSelectionGA:
    """
    FeaturesSelectionGA
    This class uses Genetic Algorithm to find out the best features for an input model
    using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
    used for GA but it can be changed accordingly.
    """

    def __init__(self, model, x, y, verbose = 0, ff_obj = None):
        """
        Parameters
        -----------
        model : scikit-learn supported model,
            x :  {array-like}, shape = [n_samples, n_features]
                 Training vectors, where n_samples is the number of samples
                 and n_features is the number of features.

            y  : {array-like}, shape = [n_samples]
                 Target Values
        cv_split: int
                 Number of splits for cross_validation to calculate fitness.

        verbose: 0 or 1
        """
        self.model = model
        self.n_features = x.shape[1]
        self.x = x
        self.y = y
        self.verbose = verbose
        if self.verbose == 1:
            print(
                "Model {} will select best features among {} features.".format(
                    model, x.shape[1]
                )
            )
            print("Shape od train_x: {} and target: {}".format(x.shape, y.shape))
        self.final_fitness = []             # NOTE
        self.fitness_in_generation = {}     # NOTE
        self.best_ind = None
        if ff_obj == None:
            self.env = Env()
        else:
            self.env = ff_obj

    def evaluate(self, individual: List[int]):
        fitness = self.env.calculate_fitness(individual)

        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual, fitness))

        return (fitness,)   

    def get_final_scores(self, pop, fits):
        self.final_fitness = list(zip(pop, fits))

    def generate(self, n_pop, cxpb=0.5, mutxpb=0.2, ngen=5, set_toolbox=False):

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

        pop = [init_individual(self.env.total_f_N, self.env.feature_num) for _ in range(n_pop)]
        CXPB, MUTPB, NGEN = cxpb, mutxpb, ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.env.calculate_fitness, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g + 1))
            selected_pop = select(pop, len(pop))
            self.fitness_in_generation[str(g + 1)] = max(
                [ind.fitness for ind in pop]
            )

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(selected_pop[::2], selected_pop[1::2]):
                if random.random() < CXPB:
                    cross_over(child1, child2)
                    child1.fitness = None
                    child2.fitness = None

            for mutant in selected_pop:
                if random.random() < MUTPB:
                    mutate(mutant, self.env.total_f_N)
                    mutant.fitness = None

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in selected_pop if not ind.fitness]
            fitnesses = list(map(self.env.calculate_fitness, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness = fit 
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = selected_pop                             

            # Gather all the fitnesses in one list and print the stats
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

        print("-- Only the fittest survives --")

        self.best_ind = max(pop, key = lambda ind: ind.fitness)
        print(
            "Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness)
        )
        self.get_final_scores(pop, fits)

        return pop

    def save_fitness_in_generation(self, file_name: str):
        """
            Save fitness in each generation in a file
        """
        joblib.dump(self.fitness_in_generation, file_name)