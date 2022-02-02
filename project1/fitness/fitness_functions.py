from abc import ABC, abstractmethod

import numpy as np


class FitnessFunction(ABC):
    pass


class RealValueFitnessFunction(FitnessFunction):
    def __init__(self,
                 interval: tuple[int, int] = (0, 16),
                 target_interval: tuple[int, int] = None,
                 distance_factor: float = 0.05
                 ):
        self.interval = interval
        self.target_interval = target_interval
        self.distance_factor = distance_factor

    @classmethod
    def bits_to_num(cls, bits: np.ndarray) -> np.ndarray:
        """Converts Nxb array of bits into a Nx1 vector of numbers.

        This is done by taking the dot product with a 2-power vector.

        :param bits:A (Nxb) numpy array of bits.
                    N: number of rows of bits, b: number of bits in each row.
        :return: A (Nx1) numpy array containing the number each of the rows represent
        """
        return np.dot(bits, 2 ** np.arange(np.atleast_2d(bits).shape[1])[::-1])

    @classmethod
    def scale_nums(cls, nums: np.ndarray, from_interval: tuple[int, int], to_interval: tuple[int, int]) -> np.ndarray:
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

    @abstractmethod
    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population using the population's phenomes (real values).

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness of the population.
        """

        raise NotImplementedError('Subclasses must implement fitness_function()')

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

class SineFitness(RealValueFitnessFunction):
    """Fitness function using sin(x)."""

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to sin(x).

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (sin(x)) of the population.
        """

        return np.sin(phenomes) + 1  # Add 1 to force non-negative fitness values


class SquaredFitness(RealValueFitnessFunction):
    """Fitness function using x**2"""

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to x**2.

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (x**2) of the population.
        """
        return phenomes ** 2


class CubedFitness(RealValueFitnessFunction):
    """Fitness function using x**3."""

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to x**3

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (x**3) of the population.
        """

        return phenomes ** 3


class PolynomialFitness(RealValueFitnessFunction):
    """Fitness function using cos(x)."""
    def __init__(self, interval: tuple[int, int], a: float, b: float, c: float):
        super().__init__(interval)
        self.a = a
        self.b = b
        self.c = c

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to the polynomial ax**2 + bx + c.

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (ax**2 + bx + c) of the population.
        """

        return self.a * phenomes ** 2 + self.b * phenomes + self.c


class CosineFitness(RealValueFitnessFunction):
    """Fitness function using cos(x)."""

    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to cos(x)

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (x**3) of the population.
        """

        return np.cos(phenome) + 1  # Add 1 to force non-negative fitness values


class LinearWithSineFitness(RealValueFitnessFunction):
    """Fitness function using cos(x)."""
    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to the polynomial ax**2 + bx + c.

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (ax**2 + bx + c) of the population.
        """

        return phenomes + np.sin(phenomes) + 1


if __name__ == '__main__':
    p = np.array([
        [0, 0, 0, 0],  # 0
        [0, 0, 1, 1],  # 3
        [0, 1, 1, 0],  # 6
        [0, 1, 1, 1],  # 7
        [1, 0, 1, 0],  # 10
        [1, 1, 0, 0],  # 12
        [1, 1, 1, 1],  # 15
    ])
    p2 = np.array([0, 0, 0, 0])
    x2 = SquaredFitness()
    sin = SineFitness()
    print("x2: ", x2.fitness(p))
    print("sin: ", sin.fitness(p))

    print("x2: ", x2(p))
    print("sin: ", sin(p))
    print("x2: ", x2(p[0]))
    print("sin: ", sin(p[0]))
