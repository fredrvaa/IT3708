from abc import ABC, abstractmethod

import numpy as np

from fitness.LinReg import LinReg
from data_utils.dataset import Dataset


class FitnessFunction(ABC):
    """Base class used to standardize the interface for implemented fitness functions."""

    def __init__(self, maximizing: bool = True):
        """
        :param maximizing: Whether of not the fitness function should be maximized, or minimized.
        """
        self.maximizing = maximizing

    @abstractmethod
    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population using the population's phenomes (real values).

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        raise NotImplementedError('Subclasses must implement fitness_function()')

    @abstractmethod
    def __call__(self, population: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population using the population's genomes.

        :param population:  A (NxC) numpy array of a population's genomes.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        raise NotImplementedError('Subclasses must implement __call__()')


class RealValueFitnessFunction(FitnessFunction):
    """Base class for implementing fitness functions using real values (sin, cos, x**2 etc.)."""

    def __init__(self,
                 interval: tuple[int, int] = (0, 16),
                 target_interval: tuple[int, int] = None,
                 distance_factor: float = 0.05,
                 *args, **kwargs):
        """
        :param interval: Interval the real value function should function on.
        :param target_interval: Target interval the solution should be in. (Fitness is penalized outside this interval).
        :param distance_factor: Specifies how hard the fitness should be penalized for being outside target_interval.
        """

        super().__init__(*args, **kwargs)
        self.interval = interval
        self.target_interval = target_interval
        self.distance_factor = distance_factor

    @staticmethod
    def bits_to_num(bits: np.ndarray) -> np.ndarray:
        """Converts Nxb array of bits into a Nx1 vector of numbers.

        This is done by taking the dot product with a 2-power vector.

        :param bits:A (Nxb) numpy array of bits.
                    N: number of rows of bits, b: number of bits in each row.
        :return: A (Nx1) numpy array containing the number each of the rows represent
        """

        return np.dot(bits, 2 ** np.arange(np.atleast_2d(bits).shape[1])[::-1])

    @staticmethod
    def scale_nums(nums: np.ndarray, from_interval: tuple[int, int], to_interval: tuple[int, int]) -> np.ndarray:
        """Scales a numpy array of nums from one interval to the another

        :param nums: A (Nx1) numpy array of numbers.
        :param from_interval: A tuple (low, high) of the original interval.
        :param to_interval: A tuple (low, high) of the interval the numbers should be scaled to.
        :return: A scaled (Nx1) numpy array of numbers.
        """

        return nums * (to_interval[1] - to_interval[0]) / (from_interval[1] - from_interval[0]) + to_interval[0]

    def bits_to_scaled_nums(self, bits: np.ndarray):
        """Converts Nxb array of bits into a Nx1 vector of scaled numbers.

        The numbers are scaled to be in the interval specified in the constructor.

        :param bits: A (Nxb) numpy array of bits.
                     N: number of rows of bits, b: number of bits in each row.
        :return: A scaled (Nx1) numpy array of numbers.
        """

        # Dot product with 2-powered array to decode the bits into a real value.
        nums = self.bits_to_num(bits)

        # Scale interval
        original_interval = (0, 2 ** np.atleast_2d(bits).shape[1] - 1)

        return self.scale_nums(nums, from_interval=original_interval, to_interval=self.interval)

    def _distance_penalty(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates and returns penalty based on distance from the target interval.

        :param phenomes: A (Nx1) numpy array of numbers.
        :return: A (Nx1) numpy array consisting of distance penalties.
        """

        distance_penalties = []
        for phenome in np.atleast_1d(phenomes):
            distance_penalty = 0
            if phenome > self.target_interval[1]:
                distance_penalty = abs(phenome - self.target_interval[1])
            elif phenome < self.target_interval[0]:
                distance_penalty = abs(phenome - self.target_interval[1])
            distance_penalties.append(self.distance_factor * distance_penalty)
        return np.array(distance_penalties)

    @abstractmethod
    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population using the population's phenomes (real values).

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        raise NotImplementedError('Subclasses must implement fitness_function()')

    def __call__(self, population: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population using the population's genomes (bits).

        :param population:  A (Nxb) numpy array a population's genomes.
                            N: number of individuals in population, b: number of bits representing each individual.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        phenomes = self.bits_to_scaled_nums(population)

        if self.target_interval is not None:
            fitness = self.fitness(phenomes) - self._distance_penalty(phenomes)
        else:
            fitness = self.fitness(phenomes)

        fitness = fitness.clip(min=0)
        return fitness


class SineFitness(RealValueFitnessFunction):
    """Fitness function using sin(x)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fitness(self, phenomes: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Calculates fitness of a population according to sin(x).

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (sin(x)) of the population.
        """

        return np.sin(phenomes) + 1  # Add 1 to force non-negative fitness values


class LinearRegressionFitness(FitnessFunction):
    """Fitness function using the provided LinReg file."""

    def __init__(self, dataset: Dataset, *args, **kwargs):
        """
        :param dataset: A Dataset object to perform linear regression on.
        """
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    @staticmethod
    def bitlist_to_bitstring(bitlist: np.ndarray):
        """Converts a numpy array of int to str.

        :param bitlist: Numpy array consisting of integers.
        :return: Numpy array consisting of strings.
        """

        return bitlist.astype(str)

    def fitness(self, phenomes: list[np.ndarray]) -> np.ndarray:
        """Calculates fitness of a population using the provided LinReg file.

        The phenomes here are feature vectors selected by selecting features using a bitstring.

        :param phenomes: A list of N feature vectors.
                         N: number of individuals in the population.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        fitness = []
        for phenome in phenomes:
            if np.count_nonzero(phenome) == 0:
                fitness.append(0)
            else:
                # print("x:", phenome)
                # print("y:", self.dataset.y)
                fitness.append(LinReg().get_fitness(phenome, self.dataset.y))
        return np.asarray(fitness)

    def __call__(self, population: np.ndarray):
        """Calculates fitness of a population using the population's genomes (bits).

        Genomes(bits/bitstrings) are decoded into phenomes(feature vectors) by applying the bitstrings to the full
        feature vector.

        Example: Genome: [1,0,1], Feature: [0.3,0.2,0.1] -> Phenome: [0.3,0.1]

        Fitness is then calculated for each phenome using the provided LinReg file.

        :param population:  A (Nxb) numpy array a population's genomes.
                            N: number of individuals in population, b: number of bits representing each individual.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        phenomes = []
        for genome in np.atleast_2d(population):
            phenomes.append(LinReg().get_columns(self.dataset.x, self.bitlist_to_bitstring(genome)))
        return self.fitness(phenomes)

