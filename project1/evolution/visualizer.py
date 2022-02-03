import matplotlib.pyplot as plt

from evolution.genetic_algorithm import GeneticAlgorithm


class Visualizer:
    def __init__(self, fitted_algorithms: list[GeneticAlgorithm]):
        self.fitted_algorithms = fitted_algorithms

        plt.ioff()

    def visualize_fitness(self):
        """Visualizes fitness metrics from the last fit.

        For each generation the sum, max/min, and mean of fitness over the whole generation is plotted.
        """

        fig, ax = plt.subplots(1, 3, figsize=(12, 12))
        max_or_min = 'Max' if self.fitted_algorithms[0].fitness_function.maximizing else 'Min'
        for i, f in enumerate(['Sum', max_or_min, 'Mean']):
            ax[i].set_title(f)
            ax[i].set_xlabel('Generation')

            for algo in self.fitted_algorithms:
                ax[i].plot(algo.fitness_history[:, i], label=algo.__class__.__name__)

            if i == 0:
                ax[i].set_ylabel(f'Fitness')
                ax[i].legend()

        plt.show()

    def visualize_entropy(self):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title('Entropy')
        ax.set_xlabel('Generation')
        ax.set_ylabel(f'Entropy')
        for algo in self.fitted_algorithms:
            ax.plot(algo.entropy_history, label=algo.__class__.__name__)

        ax.legend()
        plt.show()
