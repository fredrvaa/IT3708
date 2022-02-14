import argparse
import os

from evolution.genetic_algorithm import GeneticAlgorithm
from evolution.visualizer import Visualizer
from fitness.LinReg import LinReg
from fitness.fitness_functions import LinearRegressionFitness
from data_utils.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, required=True, help='path/to/fitted/algorithm')
args = parser.parse_args()

# Load dataset and get baseline
dataset: Dataset = Dataset.load('dataset.txt')
baseline = LinReg().get_fitness(dataset.x, dataset.y)

# Load fitted algorithm and get fitness masking out the features
algo: GeneticAlgorithm = GeneticAlgorithm.load(args.algorithm)
bitlist = algo.fittest_individual()
bitstring = LinearRegressionFitness.bitlist_to_bitstring(bitlist)

columns = LinReg().get_columns(dataset.x, bitstring)
fitness = LinReg().get_fitness(columns, dataset.y)

print(f'RMSE with all features: {baseline}')
print(f'RMSE by masking out features: {fitness}')
print(f'Mask: {bitstring}')