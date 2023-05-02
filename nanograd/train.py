import numpy as np
from mlp import MLP
from engine import Value
import matplotlib.pyplot as plt

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
BATCH_SIZE = 10
loss_l = []
x = np.linspace(0,10*np.pi, N)
y = x*2 #np.sin(x)


mlp = MLP((SEQ_SIZE,10,10,SEQ_SIZE))
optim = SGD(mlp.parameters(), 0.0001)

for i in range(N_ITER):
    idx = np.random.randint(N, size=SEQ_SIZE)
    #idx = np.random.randint(N, size=BATCH_SIZE)
    #inputs = [[i] for i in x[idx]]

    #out = list(map(mlp, inputs))
    #out = [i for j in out for i in j]
    out=mlp(x[idx])
    mlp.zero_grad()
    loss = mse(out, y[idx])

    loss.backward()
    optim.step()

    print(loss)
    loss_l.append(loss.data)

plt.plot(np.log(loss_l))
plt.show()

plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")
print([out.data for out in mlp(x[0:SEQ_SIZE])])
plt.plot(x[0:SEQ_SIZE], [out.data for out in mlp(x[0:SEQ_SIZE])], label="pred")
plt.legend()
plt.show()

