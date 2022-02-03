from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Dataclass for loading and storing a dataset."""
    x: np.ndarray
    y: np.ndarray

    @staticmethod
    def load(file_name: str) -> 'Dataset':
        """Loads dataset from file

        :param file_name: Path to file from current directory.
        :return: A Dataset object.
        """
        data = pd.read_csv(file_name, sep=',', header=None)
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        return Dataset(x.to_numpy(), y.to_numpy())