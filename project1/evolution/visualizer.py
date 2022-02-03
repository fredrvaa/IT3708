import matplotlib.pyplot as plt

from evolution.genetic_algorithm import GeneticAlgorithm


class Visualizer:
    """Utility class used to visualize plots from fitting genetic algorithms."""

    def __init__(self, fitted_algorithms: list[GeneticAlgorithm]):
        self.fitted_algorithms = fitted_algorithms

        plt.ioff()

    def visualize_fitness(self, baseline: float = None):
        """Visualizes fitness metrics from the last fit of all fitted_algorithms passed in the constructor.

        For each generation the sum, max/min, and mean of fitness over all generations are plotted.
        """

        fig, ax = plt.subplots(1, 3, figsize=(12, 12))
        max_or_min = 'Max' if self.fitted_algorithms[0].fitness_function.maximizing else 'Min'
        for i, f in enumerate(['Sum', max_or_min, 'Mean']):
            ax[i].set_title(f)
            ax[i].set_xlabel('Generation')

            for algo in self.fitted_algorithms:
                ax[i].plot(algo.fitness_history[:, i], label=algo.__class__.__name__)

            if baseline is not None and i != 0:  # Don't plot baseline on sum plot
                ax[i].axhline(y=baseline, color='r', linestyle='--', label='Baseline')

            if i == 0:
                ax[i].set_ylabel(f'Fitness')
                ax[i].legend()

        plt.show()

    def visualize_entropy(self):
        """Visualizes entropy from the last fit of all fitted_algorithms passed in the constructor."""

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title('Entropy')
        ax.set_xlabel('Generation')
        ax.set_ylabel(f'Entropy')
        for algo in self.fitted_algorithms:
            ax.plot(algo.entropy_history, label=algo.__class__.__name__)

        ax.legend()
        plt.show()
