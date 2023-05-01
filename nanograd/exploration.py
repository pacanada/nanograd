
from __future__ import annotations
import math

from graph_utils import draw_dot

class Value:
    def __init__(self, data: float, _children=(), _op="", label=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0 # init at 0, not none
        self.label=label
        self._backprop=lambda _: None

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def __add__(self, other: Value)-> Value:
        def _backprop(self):
            for child in self._prev:
                child.grad += self.grad

        out = Value(self.data + other.data, (self,other), _op="+")
        out._backprop = _backprop

        return out
    
    def __mul__(self, other: Value)->Value:
        def _backprop(self):
            # can we just substitute _prev[0] for self??
            self._prev[0].grad += self.grad*self._prev[1].data
            self._prev[1].grad += self.grad*self._prev[0].data
        out = Value(self.data*other.data, (self, other), _op="*")
        out._backprop = _backprop
        return out
    
    def __pow__(self, other):
        def _backprop(self):
            # TODO: fix, a bit complex to follow dc/da = b*a^(b-1) being c = a^b
            self._prev[0].grad += self.grad*self._prev[1].data*self._prev[0].data**(self._prev[1].data-1)
            # dc/db = a^b log(a) being c=a^b
            #self._prev[1].grad += self.grad*self._prev[0].data**self._prev[1].data*math.log(self._prev[0].data)
        out = Value(self.data**other.data, (self, other), _op=f"**{other.data}")
        out._backprop = _backprop
        return out
    
    def __truediv__(self, other):
        return self*other**Value(-1)
    def __sub__(self, other):
        return self+other*Value(-1)

    
    def relu(self)-> Value:
        raise NotImplementedError
    
    def backward(self, last_node = True):
        if last_node:
            self.grad = 1

        self._backprop(self)

        for value in self._prev:
            value.backward(last_node=False)
    

if __name__=="__main__":

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
