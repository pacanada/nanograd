from __future__ import annotations


class Value:
    def __init__(self, data: float, _children=(), _op="", label=""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.grad = 0 # init at 0, not none
        self.label=label
        self._backprop=lambda _: None
    
    def _convert_to_value(self, other: int | float | Value) -> Value:
        if isinstance(other, int | float):
            out = Value(data=other)
        elif isinstance(other, Value):
            out = other
        else:
            raise TypeError(f"{type(other)} cannot be converted to Value")
        return out


    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def __add__(self, other: int | float | Value)-> Value:
        other = self._convert_to_value(other)
        def _backprop(self):
            for child in self._prev:
                child.grad += self.grad

        out = Value(self.data + other.data, (self,other), _op="+")
        out._backprop = _backprop

        return out
    
    def __mul__(self, other: int | float | Value)->Value:
        other = self._convert_to_value(other)
        def _backprop(self):
            # can we just substitute _prev[0] for self??
            self._prev[0].grad += self.grad*self._prev[1].data
            self._prev[1].grad += self.grad*self._prev[0].data
        out = Value(self.data*other.data, (self, other), _op="*")
        out._backprop = _backprop
        return out
    
    def __pow__(self, other: int | float | Value)-> Value:
        other = self._convert_to_value(other)
        def _backprop(self):
            # TODO: fix, a bit complex to follow dc/da = b*a^(b-1) being c = a^b
            self._prev[0].grad += self.grad*self._prev[1].data*self._prev[0].data**(self._prev[1].data-1)
            # dc/db = a^b log(a) being c=a^b
            #self._prev[1].grad += self.grad*self._prev[0].data**self._prev[1].data*math.log(self._prev[0].data)
        out = Value(self.data**other.data, (self, other), _op=f"**{other.data}")
        out._backprop = _backprop
        return out
    
    def __truediv__(self, other: int | float | Value)-> Value:
        other = self._convert_to_value(other)
        return self*other**-1
    def __sub__(self, other:int | float | Value)-> Value:
        return self+other*-1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other + self
        return self+other*-1

    
    def relu(self)-> Value:
        out = Value(self.data if self.data > 0 else 0, (self, ), _op="ReLU")
        def _backprop(self):
            # TODO: fix, a bit complex to follow dc/da = b*a^(b-1) being c = a^b
            self._prev[0].grad += self.grad if self.data > 0 else 0

        out._backprop = _backprop
        return out
    
    def backward(self, last_node = True) -> None:
        if last_node:
            self.grad = 1

        self._backprop(self)

        for value in self._prev:
            value.backward(last_node=False)
