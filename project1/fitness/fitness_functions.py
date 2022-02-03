from abc import ABC, abstractmethod

import numpy as np

from fitness.LinReg import LinReg
from data_utils.dataset import Dataset


class FitnessFunction(ABC):
    def __init__(self, maximizing: bool = True):
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
    def __init__(self,
                 interval: tuple[int, int] = (0, 16),
                 target_interval: tuple[int, int] = None,
                 distance_factor: float = 0.05,
                 *args, **kwargs):
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
        # Dot product with 2-powered array to decode the bits into a real value.
        nums = self.bits_to_num(bits)

        # Scale interval
        original_interval = (0, 2 ** np.atleast_2d(bits).shape[1] - 1)

        return self.scale_nums(nums, from_interval=original_interval, to_interval=self.interval)

    def _distance_to_target(self, phenomes: np.ndarray) -> np.ndarray:
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
            fitness = self.fitness(phenomes) - self._distance_to_target(phenomes)
        else:
            fitness = self.fitness(phenomes)

        fitness = fitness.clip(min=0)
        return fitness


class LinearRegressionFitness(FitnessFunction):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset

    @staticmethod
    def bitlist_to_bitstring(bitlist: np.ndarray):
        return bitlist.astype(str)

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
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
        phenomes = []
        for genome in np.atleast_2d(population):
            phenomes.append(LinReg().get_columns(self.dataset.x, self.bitlist_to_bitstring(genome)))
        return self.fitness(phenomes)

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


if __name__ == '__main__':
    d = Dataset(x=np.array([[1,1,0],
                            [0,1,1]]),
                y=np.array([2,1]))

    lf = LinearRegressionFitness(dataset=d)

    pop = np.array([[1,1,0],[0,1,0],[0,0,1],[0,0,0]])
    print(lf(pop))

