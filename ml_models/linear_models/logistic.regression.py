"""
Binary logistic regression family: unregularized, L2-regularized
(ridge-style), and L1-regularized (lasso-style) variants.

Every model in this module predicts ``P(y = 1 | x) = sigmoid(X @ w)``
(with an optional intercept term folded into `w` via a prepended bias
column) and is fit by minimizing the binary cross-entropy loss. The
implementation is built exclusively on the math_foundations library
(`Vector`, `Matrix`, `Optimization`, `InformationTheory`) and the
`ml_models` core utilities (`MLModel`, `accuracy`, `sigmoid`). No
third-party numerical package is imported.

`LogisticRegressionL1`'s proximal (soft-thresholding) update is reused
from `linear_models.linear_regression`, since the operator is
mathematically identical regardless of the loss function it is applied
to (only the gradient term differs between lasso regression and
L1-regularized logistic regression).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Union

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from math_foundations.Vectors.vector import Vector 
from math_foundations.Matrix.matrix import Matrix  
from math_foundations.Optimization.base import Optimizer 
from math_foundations.Optimization.first_order import Adam  
from math_foundations.InformationTheory.information_theory import binary_cross_entropy  
from ml_models.base_class import MLModel  
from ml_models.metrics import accuracy  
from ml_models.activations import sigmoid  
from ml_models.linear_models.linear_regression import _soft_threshold  # type: ignore

Parameter = Union[float, Vector, Matrix]
