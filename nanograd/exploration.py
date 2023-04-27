
from __future__ import annotations

from graph_utils import draw_dot
class Value:
    def __init__(self, data: float, _children=(), _op=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = None

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def __add__(self, other: Value)-> Value:
        return Value(self.data + other.data, (self,other), _op="add")
    
    def __mul__(self, other: Value)->Value:
        return Value(self.data*other.data, (self, other), "mul")
    
    def backward(self):
        if self.grad is None:
            self.grad = 1
        for value in self._prev:
            value.backward()



a = Value(1.0)
b = Value(2.0)
c = Value(3.0)

d = a + b + c

graph = draw_dot(d)
graph.render()
