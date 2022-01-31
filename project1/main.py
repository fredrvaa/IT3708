import numpy as np

from evolution.genetic_algorithm import GeneticAlgorithm
from fitness.LinReg import LinReg
from fitness.fitness_functions import SineFitness, SquaredFitness, LinearWithSineFitness
from data_utils.dataset import Dataset

dataset = Dataset.read_file('dataset.txt')
#model = LinReg()

fitness_function = LinearWithSineFitness(interval=(0, 128))

ga = GeneticAlgorithm(
    population_size=32,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.6,
    p_mutation=0.001
)

ga.fit(generations=20, verbose=True, visualize=True)
ga.visualize_fitness()

#print(model.get_fitness(dataset.x, dataset.y))