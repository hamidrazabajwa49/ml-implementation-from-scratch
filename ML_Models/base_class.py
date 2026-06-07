import os
import sys
from typing import List, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Vectors.vector import Vector
from Matrix.matrix import Matrix


class MLModels:

    def fit(self, X: Matrix, y: Vector) -> None:
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X: Matrix) -> Vector:
        raise NotImplementedError("Subclasses must implement predict()")

    def score(self, X: Matrix, y: Vector) -> float:
        raise NotImplementedError("Subclasses must implement score()")

    def parameters(self) -> List[Union[float, Vector, Matrix]]:
        raise NotImplementedError("Subclasses must implement parameters()")

    def _validate_Xy(self, X: Matrix, y: Vector) -> None:
        if not isinstance(X, Matrix):
            raise TypeError(f"X must be a Matrix, got {type(X).__name__}")
        if not isinstance(y, Vector):
            raise TypeError(f"y must be a Vector, got {type(y).__name__}")
        if X.n_rows != len(y):
            raise ValueError(
                f"X and y must have the same number of samples: "
                f"X has {X.n_rows} rows but y has {len(y)} elements"
            )
        if X.n_rows == 0:
            raise ValueError("X and y must not be empty")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, 'w'):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted. Call fit() before predict() or score()."
            )

    def _add_bias_column(self, X: Matrix) -> Matrix:
        new_rows = []
        for i in range(X.n_rows):
            row_components = [1.0] + list(X.rows[i].components)
            new_rows.append(row_components)
        return Matrix(new_rows)
