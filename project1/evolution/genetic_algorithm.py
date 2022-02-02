import time
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from project1.fitness.fitness_functions import RealValueFitnessFunction


class GeneticAlgorithm(ABC):
    def __init__(self,
                 population_size: int = 32,
                 n_bits: int = 8,
                 fitness_function=None,
                 maximize: bool = True,
                 p_cross_over: float = 0.6,
                 p_mutation: float = 0.05,
                 offspring_multiplier: int = 1):
        """
        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :param fitness_function: Fitness function to use during evolution.
        :param maximize: Whether or not to maximize fitness (otherwise minimize).
        :param p_cross_over: Probability of crossover of two parents.
        :param p_mutation: Probability of mutating offspring.
        :param offspring_multiplier: Decides how many offspring are created in each generation:
                                     population_size is selected from population_size * offspring_multiplier offsprings
        """

        self.population_size: int = population_size
        self.n_bits: int = n_bits

        if fitness_function is None:
            raise TypeError('fitness_function must be specified')

        self.fitness_function = fitness_function
        self.maximize: bool = maximize
        self.p_cross_over: float = p_cross_over
        self.p_mutation: float = p_mutation
        self.offspring_multiplier = offspring_multiplier

        self.population: np.ndarray = self.init_population(population_size, n_bits)

        # Used to store histories during fit
        self.population_history: list[np.ndarray] = []
        self.fitness_history: list[np.ndarray] = []
        self.entropy_history: list[float] = []

    @staticmethod
    def init_population(population_size: int, n_bits: int) -> np.ndarray:
        """Initializes population of size (n_individuals x n_bits).

        Assignment task a).

        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :return: Numpy array of the whole population. Shape: (population_size x n_bits).
        """

        return np.random.randint(0, 2, (population_size, n_bits))

    @staticmethod
    def calculate_entropy(population: np.ndarray, epsilon=1e-12) -> float:
        probabilities = population.mean(axis=0)
        probabilities = probabilities.clip(min=epsilon)
        return -np.dot(probabilities, np.log2(probabilities))

    def _get_fitness_stats(self) -> np.ndarray:
        """Calculates fitness of whole population and returns sum, max, and mean of these.

        :return: A (3x1) numpy array of the sum, max, and mean of the fitness of the population.
        """

        fitness: np.ndarray = self.fitness_function(population=self.population)
        return np.array([fitness.sum(), fitness.max(), fitness.mean()])

    def _get_selection_probabilities(self, fitness: np.ndarray) -> np.ndarray:
        """Calculates selection probabilities from a fitness vector.

        The probabilities are calculated using the roulette wheel method.

        :param fitness: A (Nx1) numpy array specifying the fitness of each individual in the population.
        :return: A (Nx1) numpy array of the probabilities that an individual will be chosen as a parent.
        """

        return fitness / fitness.sum()

    def _parent_selection(self, population: np.ndarray):
        """Selects parents for the next generation of the population.

        :return: A multiset chosen from the current population.
        """
        parent_population = population.copy()

        fitness = self.fitness_function(population=parent_population)
        probabilities = self._get_selection_probabilities(fitness)
        indeces = np.random.choice(len(fitness),
                                   size=self.population_size,
                                   replace=True,
                                   p=probabilities)

        parent_population = parent_population[indeces]
        np.random.shuffle(parent_population)  # Shuffles the mating pool
        return parent_population

    def _cross_over(self, popultation: np.ndarray) -> np.ndarray:
        """Performs cross over for a whole population.

        Two and two individuals are crossed. If the population contains an odd
        number of individuals, the last one is not crossed, and just passes through.

        This should be done after selection.

        Assignment task c)

        :param popultation: A (Nxb) numpy array of a population.
        :return: A (Nxb) numpy array of the crossed over population.
        """

        crossed_population = popultation.copy()
        for p1, p2 in zip(crossed_population[::2], crossed_population[1::2]):
            if np.random.random() < self.p_cross_over:
                c = np.random.randint(1, self.n_bits)  # Random cross over point
                temp1 = p1[c:].copy()
                temp2 = p2[c:].copy()
                p1[c:], p2[c:] = temp2, temp1
        return crossed_population

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Mutates a population by randomly flipping bits.

        This should be done after cross over.

        Assignment task c)

        :param population: A (Nxb) numpy array of a population.
        :return: A (Nxb) numpy array of the mutated population.
        """

        mutated_population = population.copy()
        mask = np.random.choice([0, 1], size=population.shape, p=[1-self.p_mutation, self.p_mutation])
        idx = np.where(mask == 1)
        mutated_population[idx] = 1 - mutated_population[idx]
        return mutated_population


    @abstractmethod
    def _survivor_selection(self, parents: np.ndarray, offspring: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Subclass must implement _survivor_selection() method.')

    def fit(self,
            generations: int = 100,
            termination_fitness: float = None,
            verbose: bool = False,
            visualize: bool = False,
            vis_sleep: float = 0.1,
            ) -> None:
        """Fits the population through a generational loop.

        For each generation the following is done:
        1. Selection
        2. Cross over
        3. Mutation

        :param generations: Number of generations the algorithm should run.
        :param termination_fitness: Fitting stops if termination_fitness has been reached.
                                    If None, all generations are performed.
        :param verbose: Whether or not additional data should be printed during fitting.
        :param visualize: Whether or not to visualize population during fitting.
        :param vis_sleep: Sleep timer between each generation. Controls speed of visualization.
        """

        # Only visualize if the fitness function is a real value fitness function.
        # Not visualizing for regression tasks.
        visualize = visualize and issubclass(self.fitness_function, RealValueFitnessFunction)

        self.population_history = []
        self.fitness_history = []
        self.entropy_history = []

        if visualize:
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 12))
            interval = self.fitness_function.interval
            x_func = np.linspace(interval[0], interval[1], 10*(interval[1] - interval[0]))
            y_func = self.fitness_function.fitness(x_func)
            ax.plot(x_func, y_func)
            ax.set_xlabel('Value')
            ax.set_ylabel('Fitness')
            points, = ax.plot(x_func, y_func, 'ro')

        print(self.__class__.__name__)
        for g in range(generations):
            print(f'Generation {g}')
            fitness_stats = self._get_fitness_stats()
            self.fitness_history.append(fitness_stats)

            entropy = self.calculate_entropy(self.population)
            self.entropy_history.append(entropy)

            if termination_fitness is not None and fitness_stats[-1] >= termination_fitness:  # Checks mean fitness
                break
            if verbose:
                print(f'Entropy: {round(entropy, 2)}')
                fitness_table = PrettyTable(['Sum', 'Max', 'Mean'], title='Fitness')
                fitness_table.add_row([round(s, 2) for s in fitness_stats])
                print(fitness_table)
            if visualize:
                ax.set_title(f'Generation {g}')
                x = self.fitness_function.bits_to_scaled_nums(self.population)
                y = self.fitness_function.fitness(x)
                points.set_xdata(x)
                points.set_ydata(y)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(vis_sleep)

            self.population_history.append(self.population.copy())
            parents = self._parent_selection(self.population)
            crossed = self._cross_over(parents)
            mutated = self._mutate(crossed)
            self.population = self._survivor_selection(parents, mutated)

        self.population_history = np.asarray(self.population_history)
        self.fitness_history = np.asarray(self.fitness_history)
        self.entropy_history = np.asarray(self.entropy_history)


class SimpleGeneticAlgorithm(GeneticAlgorithm):
    def _survivor_selection(self, parents: np.ndarray, offspring: np.ndarray) -> np.ndarray:
        return offspring


class GeneralizedCrowding(GeneticAlgorithm):
    def __init__(self, scaling_factor: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._scaling_factor = scaling_factor

    @staticmethod
    def hamming_distance(a1: np.ndarray, a2: np.ndarray) -> int:
        return np.count_nonzero(a1 != a2)

    def _competition(self, parent: np.ndarray, offspring: np.ndarray) -> np.ndarray:
        offspring_fitness = self.fitness_function(offspring)
        parent_fitness = self.fitness_function(parent)
        if offspring_fitness > parent_fitness:
            offspring_probability = offspring_fitness / (offspring_fitness + self._scaling_factor * parent_fitness)
            return offspring if np.random.random() < offspring_probability else parent
        elif offspring_fitness < parent_fitness:
            scaled_offspring_fitness = self._scaling_factor * offspring_fitness
            offspring_probability = scaled_offspring_fitness / (scaled_offspring_fitness + parent_fitness)
            return offspring if np.random.random() < offspring_probability else parent
        else:
            return offspring if np.random.random() < 0.5 else parent

    def _survivor_selection(self, parents: np.ndarray, offspring: np.ndarray) -> np.ndarray:
        """
        Parents: [p1, p2, p3]; Offspring: [o1, o2, o3]
        Competitions: [(p1, o1)]
        :param parents:
        :param offspring:
        :return:
        """
        survivor_population = offspring.copy()
        for i, (p1, p2, o1, o2) in enumerate(zip(parents[::2], parents[1::2], offspring[::2], offspring[1::2])):
            h1 = self.hamming_distance(p1, o1) + self.hamming_distance(p2, o2)
            h2 = self.hamming_distance(p1, o2) + self.hamming_distance(p2, o1)
            if h1 < h2:  # Competitions: [(p1, o1), (p2, o2)]
                survivor_population[i*2] = self._competition(p1, o1)
                survivor_population[i*2 + 1] = self._competition(p2, o2)
            else:  # Competitions: [(p1, o2), (p2, o1)]
                survivor_population[i*2] = self._competition(p1, o2)
                survivor_population[i*2 + 1] = self._competition(p2, o1)

        return survivor_population


class DeterministicCrowding(GeneralizedCrowding):
    def __init__(self, *args, **kwargs):
        super().__init__(scaling_factor=0, *args, **kwargs)


class ProbabilisticCrowding(GeneralizedCrowding):
    def __init__(self, *args, **kwargs):
        super().__init__(scaling_factor=1, *args, **kwargs)


if __name__ == '__main__':
    def f(p):
        return p.sum()

    print(GeneticAlgorithm.calculate_entropy(np.array([[1,0,1],[0,0,0], [1,1,0]])))
