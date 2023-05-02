
import random
import numpy as np
from engine import Value

np.random.seed(1337)
random.seed(1337)
class Neuron:
    def __init__(self):
        self.w = Value(np.random.uniform(-1,1))
        self.b = Value(np.random.uniform(-1,1))
    def __call__(self, x):
        # including relu
        #return (self.w*x + self.b).relu()
        return (self.w*x).relu()

    def parameters(self)->list[Value]:
        #return [self.w]+[self.b]
        return [self.w]

class Layer:
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.neurons = [Neuron() for i in range(self.d_in)]

    def __call__(self, x):
        if len(x)!= self.d_in:
            raise Exception(f"The size of the input ({len(x)}) doesnt correspond to the size of the layer ({self.d_in})")
        out = [sum([n(x_i) for x_i, n in zip(x, self.neurons)]) for _ in range(self.d_out)]
        return out
    def parameters(self)->list[Value]:
        # Trick to unnest the list of params
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes: tuple):
        self.layers = [Layer(d_in=layer_size, d_out=sizes[i+1]) for i, layer_size in enumerate(sizes[:-1])]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self)->list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0