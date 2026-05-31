import math

class Vector:
    def __init__(self, components):
        self.components=components

    def __repr__(self):
        return f'Vector({self.components})'

    def __getitem__(self, index):
        return self.components[index]

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def element_wise(self, func):
        return Vector([func(x) for x in self.components])

    def element_wise_with(self, other, func):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")
        return Vector([func(a, b) for a, b in zip(self.components, other.components)])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result=[x + other for x in self]
            return Vector(result)
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        summed_components = [x + y for x, y in zip(self, other)]
        return Vector(summed_components)
    __radd__ = __add__

    def dot (self,other):
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        product = sum(x * y for x, y in zip(self, other))
        return product

    def __mul__ (self,num):
        result=[num*x for x in self]
        return Vector(result)
    __rmul__ = __mul__ 

    def norm (self,order=2):
        if order==1:
            result= sum(abs(x) for x in self)
            return result
        elif order==2:
            result= math.sqrt(self.dot(self))
            return result
        elif (order == math.inf):
            result= max(abs(x) for x in self)
            return result
        else:
            raise ValueError

    def angle(self,other):
        cos=(self.dot(other))/(self.norm() * other.norm())
        return math.degrees(math.acos(cos))

    def __sub__(self,other):
        if isinstance(other, (int, float)):
            result=[x - other for x in self]
            return Vector(result)
        if len(self) != len(other):
            raise ValueError(f"Dimension mismatch: {len(self)} vs {len(other)}")
        result = [x - y for x, y in zip(self, other)]
        return Vector(result)
    
    def __rsub__(self,other):
        if isinstance(other,(int,float)):
            result=[other-x for x in self]
            return Vector(result)

    def normalize(self):
        if self.norm() == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector([x/self.norm() for x in self])
    
    def projection_onto(self,other):
        if other.norm()==0:
            raise ValueError("Cannot normalize zero vector")
        return ((self.dot(other)/(other.norm())**2)*other)

