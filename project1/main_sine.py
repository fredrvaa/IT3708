import argparse
from typing import Optional

from evolution.genetic_algorithm import SimpleGeneticAlgorithm, DeterministicCrowding, ProbabilisticCrowding, \
    GeneralizedCrowding, FittestGeneticAlgorithm
from evolution.visualizer import Visualizer
from fitness.fitness_functions import SineFitness

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--generations', type=int, default=100, help='Number of generations to fit.')
parser.add_argument('-v', '--visualize', action='store_true', help='Flag to visualize sine fitness during training.')
parser.add_argument('-m', '--minimize', action='store_true', help='Flag to minimize fitness function instead of maximizing.')
parser.add_argument('-t', '--termination_fitness', type=float, default=None)
parser.add_argument('-i', '--interval', nargs='+', type=int, default=None, help='Optional constrained target interval.')
args = parser.parse_args()
if args.interval is not None:
    args.interval = tuple(args.interval)

fitness_function = SineFitness(interval=(0, 128),
                               target_interval=args.interval,
                               distance_factor=0.1,
                               maximizing=not args.minimize)

algos = []

algos.append(SimpleGeneticAlgorithm(
    population_size=500,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.6,
    p_mutation=0.001
))

algos.append(FittestGeneticAlgorithm(
    population_size=500,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.6,
    p_mutation=0.001
))

algos.append(GeneralizedCrowding(
    population_size=500,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.6,
    p_mutation=0.001,
    scaling_factor=0.5,
))

algos.append(DeterministicCrowding(
    population_size=500,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.6,
    p_mutation=0.001,
))

algos.append(ProbabilisticCrowding(
    population_size=500,
    n_bits=15,
    fitness_function=fitness_function,
    p_cross_over=0.7,
    p_mutation=0.01,
))

for algo in algos:
    algo.fit(generations=args.generations,
             verbose=True,
             visualize=args.visualize,
             vis_sleep=0,
             termination_fitness=args.termination_fitness)
    algo.save(f'save_data/sine/{algo.__class__.__name__}_{args.generations}.pkl')

visualizer = Visualizer(algos)
visualizer.visualize_fitness()
visualizer.visualize_entropy()
