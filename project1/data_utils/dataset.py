import os
from dataclasses import dataclass

import numpy as np
import pandas as pd



@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def read_file(cls, file_name: str) -> 'Dataset':
        data = pd.read_csv(file_name, sep=',', header=None)
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        return Dataset(x.to_numpy(), y.to_numpy())

if __name__ == '__main__':
    d = Dataset.read_file('dataset.txt')
    print(d.x.shape, d.y.shape)