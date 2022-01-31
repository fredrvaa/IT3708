import numpy as np


def _bits_to_num(bits: np.ndarray) -> np.ndarray:
    """Converts Nxb array of bits into a Nx1 vector of numbers.

    :param bits:A (Nxb) numpy array of bits.
                N: number of rows of bits, b: number of bits in each row.
    :return: A (Nx1) numpy array containing the number each of the rows represent
    """
    return np.dot(bits, 2 ** np.arange(bits.shape[1])[::-1])


def _scale_nums(nums: np.ndarray, from_interval: tuple, to_interval: tuple) -> np.ndarray:
    """Scales a numpy array of nums from one interval to the another

    :param nums: A (Nx1) numpy array of numbers.
    :param from_interval: A tuple (low, high) of the original interval.
    :param to_interval: A tuple (low, high) of the interval the numbers should be scaled to.
    :return: A scaled (Nx1) numpy array of numbers.
    """
    return nums * (to_interval[1] - to_interval[0]) / (from_interval[1] - from_interval[0]) + to_interval[0]


def x2(population: np.ndarray, interval: tuple = (0, 16)) -> np.ndarray:
    """Finds the fitness of each individual in a population where the goal is to maximize x**2.

    Each individual has its genome (bitstring) decoded into a phenome (real number value).
    Then, the phenome interval is scaled before calculating fitness.

    :param population:  A (Nxb) numpy array with a set of individuals.
                        N: number of individuals, b: number of bits in the genome.
    :param interval: A tuple (low, high) of the interval the numbers should be scaled to.
    :return: A (Nx1) numpy array containing the fitness score of each individual.
    """

    # Dot product with 2-powered array to decode the genome into a phenome.
    phenome = _bits_to_num(population)

    # Scale interval
    original_interval = (2 ** population.shape[1] - 1, 0)

    scaled_phenome = _scale_nums(phenome, from_interval=original_interval, to_interval=interval)
    return scaled_phenome ** 2


def sin(population: np.ndarray, interval: tuple = (0, 128)) -> np.ndarray:
    """Finds the fitness of each individual in a population where the goal is to maximize sin(x).

    Each individual has its genome (bitstring) decoded into a phenome (real number value) before calculating fitness.

    :param population:  A (Nxb) numpy array with a set of individuals.
                        N: number of individuals, b: number of bits in the genome.
    :param interval: A tuple (low, high) of the interval the numbers should be scaled to.
    :return: A (Nx1) numpy array containing the fitness score of each individual.
    """

    # Dot product with 2-powered array to decode the genome into a phenome.
    phenome = _bits_to_num(population)

    # Scale interval
    original_interval = (2 ** population.shape[1] - 1, 0)

    scaled_phenome = _scale_nums(phenome, from_interval=original_interval, to_interval=interval)

    return np.sin(scaled_phenome) + 1  # Add 1 to make all fitness scores non-negative


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
    print("x2: ", x2(p))
    print("sin: ", sin(p))
