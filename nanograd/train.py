import random
import numpy as np
#from mlp import MLP
#from engine import Value
import matplotlib.pyplot as plt

#import random
import numpy as np
from engine import Value
# it has to be in the same file, what is that!!??
np.random.seed(1337)
random.seed(1337)
class Neuron:
    def __init__(self, dim_in: int):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(dim_in)]
        self.b = Value(np.random.uniform(-1,1))
    def __call__(self, x: list[Value]):
        # including relu
        return sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
        #return (sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b).relu()

    def parameters(self)->list[Value]:
        return self.w+[self.b]

class Layer:
    def __init__(self, d_in: int, d_out: int):
        self.d_in = d_in
        self.d_out = d_out
        self.neurons = [Neuron(d_in) for _ in range(self.d_out)]

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
            p.grad=0

class SGD:
    def __init__(self, params: list[Value], lr: float=0.1):
        self.params = params
        self.lr = lr
    def step(self):
        for p in self.params:
            p.data -= self.lr*p.grad

def mse(output: list[Value], target: list[Value])-> Value:
    if len(output)!=len(target):
        raise Exception(f"Output size ({len(output)}) and target size ({len(target)}) does match!")
    # good old mse
    loss = sum([(t-i)**2 for t, i in zip(target, output)])/len(output)
    return loss

N = 1000
N_ITER = 100
SEQ_SIZE = 1
BATCH_SIZE = 50
loss_l = []
x = np.linspace(0,10*np.pi, N)
y = np.sin(x)


mlp = MLP((SEQ_SIZE,5,5,5,SEQ_SIZE))
optim = SGD(mlp.parameters(), 0.00001)

for i in range(N_ITER):
    #idx = np.random.randint(N, size=SEQ_SIZE)
    idx = np.random.randint(N, size=BATCH_SIZE)
    inputs = [[i] for i in x[idx]]

    out = list(map(mlp, inputs))
    out = [i for j in out for i in j]
    #out=mlp(x[idx])
    mlp.zero_grad()
    loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    #loss = mse(out, y[idx])

    loss.backward()
    optim.step()

    print(loss)
    loss_l.append(loss.data)
print("For 2 we have ", mlp([2]))
plt.plot(np.log(loss_l))
plt.show()

plt.plot(x, [i[0].data for i in list(map(mlp,[[x0] for x0 in x]))])
plt.show()

plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")
print([out.data for out in mlp(x[0:SEQ_SIZE])])
# plt.plot(x[0:SEQ_SIZE], [out.data for out in mlp(x[0:SEQ_SIZE])], label="pred")
# plt.legend()
# plt.show()



