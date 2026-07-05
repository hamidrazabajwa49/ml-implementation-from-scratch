"""
distributions.py
=================

Abstract base class shared by every probability distribution in this
package (discrete and continuous alike). Centralizes the common
interface (``mean``, ``variance``, ``std``, ``sample``, ``summary``) so
concrete distributions only need to implement what's actually specific
to them, and downstream code (e.g. hypothesis tests) can treat any
distribution polymorphically.

Example
-------
>>> class Constant(Distribution):
...     def __init__(self, value):
...         self.value = value
...     def mean(self): return self.value
...     def variance(self): return 0.0
...     def sample(self, num_samples=1, seed=None): return [self.value] * num_samples
>>> Constant(5).std()
0.0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional


class Distribution(ABC):
    """Common interface for all probability distributions in this package."""

    @abstractmethod
    def mean(self) -> float:
        """Return the distribution's mean (expected value)."""
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> float:
        """Return the distribution's variance."""
        raise NotImplementedError

    def std(self) -> float:
        """Return the standard deviation (``sqrt(variance())``).

        Concrete subclasses generally should *not* override this --
        overriding it independently of ``variance()`` is a common source
        of subtle mean/variance/std inconsistency bugs.
        """
        return math.sqrt(self.variance())

    @abstractmethod
    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[float]:
        """Draw ``num_samples`` i.i.d. samples from this distribution.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw; must be >= 1.
        seed : int, optional
            Seed for a local random generator, for reproducibility. If
            omitted, uses the shared global ``random`` module state.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def summary(self) -> str:
        """Return a one-line human-readable summary of mean/variance/std."""
        return (
            f"{self.__class__.__name__}: "
            f"mean={self.mean():.4f}, "
            f"variance={self.variance():.4f}, "
            f"std={self.std():.4f}"
        )
