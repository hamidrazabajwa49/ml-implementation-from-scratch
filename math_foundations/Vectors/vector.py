import math


class Vector:
    def __init__(self, components):
        if not hasattr(components, '__iter__'):
            raise TypeError("components must be iterable")
        components = list(components)
        for i, x in enumerate(components):
            if not isinstance(x, (int, float)):
                raise TypeError(f"component at index {i} must be int or float, got {type(x).__name__}")
        self.components = components

    def __repr__(self):
        return f"Vector({self.components})"

    def __len__(self):
        return len(self.components)

    def __getitem__(self, index):
        return self.components[index]

    def __setitem__(self, index, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"component must be int or float, got {type(value).__name__}")
        self.components[index] = value

    def __iter__(self):
        return iter(self.components)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.components == other.components

    def element_wise(self, func):
        return Vector([func(x) for x in self.components])

    def element_wise_with(self, other, func):
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        if len(self) != len(other):
            raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")
        return Vector([func(a, b) for a, b in zip(self.components, other.components)])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x + other for x in self.components])
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")
            return Vector([x + y for x, y in zip(self.components, other.components)])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x + other for x in self.components])
        if isinstance(other, Vector):
            return other.__add__(self)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Vector([x - other for x in self.components])
        if isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")
            return Vector([x - y for x, y in zip(self.components, other.components)])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Vector([other - x for x in self.components])
        if isinstance(other, Vector):
            return other.__sub__(self)
        return NotImplemented

    def __mul__(self, num):
        if not isinstance(num, (int, float)):
            raise TypeError(
                f"unsupported operand type for *: 'Vector' and '{type(num).__name__}'. "
                "Use .dot() for dot product."
            )
        return Vector([num * x for x in self.components])

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"unsupported operand type for /: 'Vector' and '{type(scalar).__name__}'"
            )
        if scalar == 0.0:
            raise ZeroDivisionError("division by zero")
        return Vector([x / scalar for x in self.components])

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError(f"dot product requires a Vector, got {type(other).__name__}")
        if len(self) != len(other):
            raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")
        return sum(x * y for x, y in zip(self.components, other.components))

    def norm(self, order=2):
        if len(self.components) == 0:
            return 0.0
        if order == 1:
            return sum(abs(x) for x in self.components)
        if order == 2:
            return math.sqrt(self.dot(self))
        if order == math.inf:
            return max(abs(x) for x in self.components)
        if isinstance(order, (int, float)) and order > 0:
            return sum(abs(x) ** order for x in self.components) ** (1.0 / order)
        raise ValueError(f"norm order must be a positive number or math.inf, got {order!r}")

    def normalize(self):
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalize a zero vector")
        return Vector([x / n for x in self.components])

    def angle(self, other):
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        denom = self.norm() * other.norm()
        if denom == 0.0:
            raise ValueError("cannot compute angle with a zero vector")
        cos = self.dot(other) / denom
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))

    def projection_onto(self, other):
        if not isinstance(other, Vector):
            raise TypeError(f"expected Vector, got {type(other).__name__}")
        n_sq = other.dot(other)
        if n_sq == 0.0:
            raise ValueError("cannot project onto a zero vector")
        return (self.dot(other) / n_sq) * other
