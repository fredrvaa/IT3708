from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int = 32,
                 n_bits: int = 8,
                 fitness_function: Callable = None,
                 fitness_interval: tuple[int, int] = (0, 128),
                 maximize: bool = True,
                 p_cross_over: float = 0.6,
                 p_mutation: float = 0.05):
        """
        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :param fitness_function: Fitness function to use during evolution.
        :param fitness_interval: Interval of the fitness function (passed to the fitness function when computing).
        :param maximize: Whether or not to maximize fitness (otherwise minimize).
        :param p_cross_over: Probability of crossover of two parents.
        :param p_mutation: Probability of mutating offspring.
        """

        self.population_size: int = population_size
        self.n_bits: int = n_bits

        if fitness_function is None:
            raise TypeError('fitness_function must be specified')

        self.fitness_function: Callable = fitness_function
        self.fitness_interval = fitness_interval
        self.maximize: bool = maximize
        self.p_cross_over: float = p_cross_over
        self.p_mutation: float = p_mutation

        self.population: np.ndarray = self._init_population(population_size, n_bits)
        self.population_history: list[np.ndarray] = []
        self.fitness_history: list[np.ndarray] = []


    def _init_population(self, population_size: int, n_bits: int) -> np.ndarray:
        """Initializes population of size (n_individuals x n_bits).

        Assignment task a).

        :param population_size: Number of individuals in population.
        :param n_bits: Number of bits used to represent each individual.
        :return: Numpy array of the whole population. Shape: (population_size x n_bits).
        """

        return np.random.random_integers(0, 1, (population_size, n_bits))

    def _get_fitness(self) -> np.ndarray:
        """Calculates fitness by passing in current population
        along with the specified interval to the fitness function.

        :return: A numpy array of the fitness of the whole population.
        """

        return self.fitness_function(population=self.population, interval=self.fitness_interval)

    def _get_fitness_stats(self) -> np.ndarray:
        """Calculates fitness of whole population and returns sum, max, and mean of these.

        :return: A (3x1) numpy array of the sum, max, and mean of the fitness of the population.
        """

        fitness: np.ndarray = self._get_fitness()
        return np.array([fitness.sum(), fitness.max(), fitness.mean()])

    def _get_selection_probabilities(self, fitness: np.ndarray) -> np.ndarray:
        """Calculates selection probabilities from a fitness vector.

        The probabilities are calculated using the roulette wheel method.

        :param fitness: A (Nx1) numpy array specifying the fitness of each individual in the population.
        :return: A (Nx1) numpy array of the probabilities that an individual will be chosen as a parent.
        """

        return fitness / fitness.sum()

    def _parent_selection(self):
        """Selects parents for the next generation of the population.

        :return: A multiset chosen from the current population.
        """

        fitness = self._get_fitness()
        probabilities = self._get_selection_probabilities(fitness)
        indeces = np.random.choice(len(self.population), size=len(self.population), replace=True, p=probabilities)
        return self.population[indeces]

    def _cross_over(self, popultation: np.ndarray) -> np.ndarray:
        """Performs cross over for a whole population.

        The population is first shuffled, then two and two individuals are crossed. If the population contains an odd
        number of individuals, the last one is not crossed, and just passes through.

        This should be done after selection.

        Assignment task c)

        :param popultation: A (Nxb) numpy array of a population.
        :return: A (Nxb) numpy array of the crossed over population.
        """

        crossed_population = popultation.copy()
        np.random.shuffle(crossed_population)  # Shuffles the mating pool
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

    def fit(self, generations: int = 100, verbose=False, visualize=False) -> None:
        """Fits the population through a generational loop.

        For each generation the following is done:
        1. Selection
        2. Cross over
        3. Mutation

        :param generations: Number of generations the algorithm should run.
        :param verbose: Whether or not additional data should be printed during fitting.
        :param visualize: Whether or not to visualize population during fitting.
        """

        self.population_history = []
        self.fitness_history = []

        for g in range(generations):
            print(f'Generation {g}')
            fitness_stats = self._get_fitness_stats()
            self.fitness_history.append(fitness_stats)
            if verbose:
                fitness_table = PrettyTable(['Sum', 'Max', 'Mean'], title='Fitness')
                fitness_table.add_row([round(s, 2) for s in fitness_stats])
                print(fitness_table)

            self.population_history.append(self.population.copy())
            parents = self._parent_selection()
            crossed = self._cross_over(parents)
            mutated = self._mutate(crossed)
            self.population = mutated

        self.population_history = np.asarray(self.population_history)
        self.fitness_history = np.asarray(self.fitness_history)

    def visualize_fitness(self):
        """Visualizes fitness metrics from the last fit.

        For each generation the sum, max, and mean of fitness over the whole generation is plotted.
        """

        fig, ax = plt.subplots(1, 3, figsize=(12, 12))
        for i, f in enumerate(['Sum', 'Max', 'Mean']):
            ax[i].plot(self.fitness_history[:, i])
            ax[i].set_title(f)
            ax[i].set_xlabel('Generation')

            if i == 0:
                ax[i].set_ylabel(f'Fitness')
        plt.show()


if __name__ == '__main__':
    ga = GeneticAlgorithm()
