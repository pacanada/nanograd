
from __future__ import annotations

from graph_utils import draw_dot
class Value:
    def __init__(self, data: float, _children=(), _op="", label=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = None
        self.label=label

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def __add__(self, other: Value)-> Value:
        return Value(self.data + other.data, (self,other), _op="add")
    
    def __mul__(self, other: Value)->Value:
        return Value(self.data*other.data, (self, other), _op="mul")
    
    def relu(self):
        raise NotImplementedError
    
    def backward(self):
        if self.grad is None:
            self.grad = 1

        if self._op == "add":
            self._prev[0].grad = self.grad
            self._prev[1].grad = self.grad
        elif self._op == "mul":
            self._prev[0].grad = self.grad*self._prev[1].data
            self._prev[1].grad = self.grad*self._prev[0].data

        for value in self._prev:
            value.backward()



a = Value(1.0, label="a")
b = Value(2.0, label="b")
c = Value(3.0, label="c")
d = Value(4.0, label="d")

e = 2*a + b + c * d
e.label="e"
e.backward()

print("The gradient of a should be:", 1)
print("The gradient of c should be:", d.data)

graph = draw_dot(e)
graph.render()
