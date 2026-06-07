import os
import sys
import math
from typing import List, Union


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from Vectors.vector import Vector
from Matrix.matrix import Matrix


class Optimizer:
    def __init__(self, lr: float = 0.01):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive")
        self.lr = lr
        self.iterations = 0
        self.history = []

    def step(self, params: List[Union[float, Vector, Matrix]],
            grads: List[Union[float, Vector, Matrix]]) -> None:
        raise NotImplementedError

    def reset(self):
        self.iterations = 0
        self.history = []

    def get_config(self) -> dict:
        return {"lr": self.lr}

    def record(self, loss: float):
        self.history.append(loss)

    def _zeros_like(self, x):
        if isinstance(x, (int, float)):
            return 0.0
        elif isinstance(x, Vector):
            return Vector([0.0] * len(x.components))
        elif isinstance(x, Matrix):
            return Matrix.zeros(x.n_rows, x.n_cols)
        else:
            raise TypeError(f"Unsupported parameter type: {type(x)}")
