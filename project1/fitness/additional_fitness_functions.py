import numpy as np

from fitness.fitness_functions import RealValueFitnessFunction


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

        return np.cos(phenomes) + 1  # Add 1 to force non-negative fitness values


class LinearWithSineFitness(RealValueFitnessFunction):
    """Fitness function using cos(x)."""
    def fitness(self, phenomes: np.ndarray) -> np.ndarray:
        """Calculates fitness of a population according to the polynomial ax**2 + bx + c.

        :param phenomes: A (Nx1) numpy array consisting of a populations phenomes.
        :return: A (Nx1) numpy array consisting of the fitness (ax**2 + bx + c) of the population.
        """

        return phenomes + np.sin(phenomes) + 1
