import argparse

from evolution.genetic_algorithm import SimpleGeneticAlgorithm, GeneralizedCrowding, \
    DeterministicCrowding, ProbabilisticCrowding, FittestGeneticAlgorithm
from evolution.visualizer import Visualizer
from fitness.LinReg import LinReg
from fitness.fitness_functions import LinearRegressionFitness
from data_utils.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--generations', type=int, default=30, help='Number of generations to fit.')
parser.add_argument('-t', '--termination_fitness', type=float, default=None)
args = parser.parse_args()

dataset = Dataset.load('dataset.txt')

baseline = LinReg().get_fitness(dataset.x, dataset.y)

fitness_function = LinearRegressionFitness(dataset=dataset, maximizing=False)

algos = []

# algos.append(SimpleGeneticAlgorithm(
#     population_size=100,
#     n_bits=dataset.x.shape[1],
#     fitness_function=fitness_function,
#     p_cross_over=0.7,
#     p_mutation=0.001
# ))

algos.append(FittestGeneticAlgorithm(
    population_size=200,
    n_bits=dataset.x.shape[1],
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001
))

algos.append(GeneralizedCrowding(
    population_size=200,
    n_bits=dataset.x.shape[1],
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
    scaling_factor=0.5,
))

algos.append(DeterministicCrowding(
    population_size=200,
    n_bits=dataset.x.shape[1],
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
))

algos.append(ProbabilisticCrowding(
    population_size=200,
    n_bits=dataset.x.shape[1],
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.001,
))

for algo in algos:
    algo.fit(generations=args.generations, verbose=True, termination_fitness=args.termination_fitness)
    algo.save(f'save_data/{algo.__class__.__name__}_reg_{args.generations}.pkl')

visualizer = Visualizer(algos)

visualizer.visualize_fitness(baseline=baseline)
visualizer.visualize_entropy()
