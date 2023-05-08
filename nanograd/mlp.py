
import numpy as np
from nanograd.engine import Value

#np.random.seed(1337)
#random.seed(1337)
class Neuron:
    def __init__(self, dim_in: int, activation:bool):
        self.activation = activation
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(dim_in)]
        self.b = Value(0)#Value(np.random.uniform(-1,1)) #Value(0)
    def __call__(self, x: list[Value]):
        # including relu
        #return sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
        if self.activation:
            return sum((wi*xi for wi, xi in zip(self.w, x)), self.b).relu()
        else:
            return sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            

    def parameters(self)->list[Value]:
        return self.w+[self.b]

class Layer:
    def __init__(self, d_in: int, d_out: int, activation:bool):
        self.d_in = d_in
        self.d_out = d_out
        self.neurons = [Neuron(d_in, activation) for _ in range(self.d_out-1)] + [Neuron(d_in, False)]

    def __call__(self, x):
        if len(x)!= self.d_in:
            raise Exception(f"The size of the input ({len(x)}) doesnt correspond to the size of the layer ({self.d_in})")
        #out = [n(x_i) for ]
        out = [n(x) for  n in self.neurons]
        return out
    def parameters(self)->list[Value]:
        # Trick to unnest the list of params
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes: tuple, activation=True):
        self.layers = [Layer(d_in=layer_size, d_out=sizes[i+1], activation=activation) for i, layer_size in enumerate(sizes[:-1])]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self)->list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad=0