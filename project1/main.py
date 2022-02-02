import numpy as np

from evolution.genetic_algorithm import SimpleGeneticAlgorithm, DeterministicCrowding, ProbabilisticCrowding, \
    GeneralizedCrowding
from evolution.visualizer import Visualizer
from fitness.fitness_functions import SineFitness, SquaredFitness, LinearWithSineFitness
from data_utils.dataset import Dataset

dataset = Dataset.read_file('dataset.txt')

fitness_function = SineFitness(interval=(0, 128), target_interval=None, distance_factor=0.1)
termination_fitness = None
algos = []

algos.append(SimpleGeneticAlgorithm(
    population_size=400,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001
))

algos.append(GeneralizedCrowding(
    population_size=400,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
    scaling_factor=0.5,
))

algos.append(DeterministicCrowding(
    population_size=400,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
))

algos.append(ProbabilisticCrowding(
    population_size=400,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
))

for algo in algos:
    algo.fit(generations=100, verbose=False, visualize=False, vis_sleep=0, termination_fitness=termination_fitness)

visualizer = Visualizer(algos)

visualizer.visualize_fitness()
visualizer.visualize_entropy()
