import argparse
import os

from evolution.genetic_algorithm import SimpleGeneticAlgorithm, GeneralizedCrowding, \
    DeterministicCrowding, ProbabilisticCrowding, GeneticAlgorithm
from evolution.visualizer import Visualizer
from fitness.LinReg import LinReg
from fitness.fitness_functions import LinearRegressionFitness
from data_utils.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, required=True, help='path/to/folder/with/fitted/algos')
args = parser.parse_args()

dataset = Dataset.load('dataset.txt')
baseline = LinReg().get_fitness(dataset.x, dataset.y)

algos = [GeneticAlgorithm.load(os.path.join(args.folder, x)) for x in os.listdir(args.folder)]

visualizer = Visualizer(algos)

visualizer.visualize_fitness(baseline=baseline)
visualizer.visualize_entropy()