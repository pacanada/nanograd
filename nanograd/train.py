import random
import numpy as np
from mlp import MLP
#from engine import Value
import matplotlib.pyplot as plt

#import random
import numpy as np
from engine import Value
from graph_utils import draw_dot
np.random.seed(1337)
random.seed(1337)


N = 1000
N_ITER = 500
SEQ_SIZE = 1
BATCH_SIZE = 30
PLOT = False
lr = 0.0001
loss_l = []
f = lambda x: x**2
x = np.linspace(0,10, N)
x_test = np.linspace(10,20, N)
y = f(x) #np.sin(x)
y_test = f(x_test)

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


mlp = MLP((SEQ_SIZE,10,10,SEQ_SIZE))
print("Number of params ", len(mlp.parameters()))
optim = SGD(mlp.parameters(), lr)

for i in range(N_ITER):
    #idx = np.random.randint(N, size=SEQ_SIZE)
    idx = np.random.randint(N, size=BATCH_SIZE)
    inputs = [[i] for i in x[idx]]

    out = list(map(mlp, inputs))
    out = [i for j in out for i in j]
    #out=mlp(x[idx])
    
    # data_loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    # alpha = 1e-4
    # reg_loss = alpha * sum((p*p for p in mlp.parameters()))
    # loss = data_loss+reg_loss
    #loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    loss = mse(out, y[idx])
    mlp.zero_grad()
    if PLOT:
        graph = draw_dot(loss, "loss_before", "svg")
        graph.render()
    loss.backward()
    #lr = 1.0 - 0.9*i/100
    for p in mlp.parameters():
        p.data -= lr * p.grad
    #optim.step()

    print(loss)
    loss_l.append(loss.data)
    if PLOT:
        graph = draw_dot(loss, "loss_after", "svg")
        graph.render()
        break
print("For 2 we have ", mlp([2]))
plt.plot(np.log(loss_l))
plt.show()


plt.plot(x, [i[0].data for i in list(map(mlp,[[x0] for x0 in x]))], label="pred")
plt.plot(x, y, label="target")
plt.plot(x_test, [i[0].data for i in list(map(mlp,[[x0] for x0 in x_test]))], label="pred")
plt.plot(x_test, y_test, label="target")
plt.legend()
plt.show()

plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")

print([out.data for out in mlp(x[0:SEQ_SIZE])])
# plt.plot(x[0:SEQ_SIZE], [out.data for out in mlp(x[0:SEQ_SIZE])], label="pred")
# plt.legend()
# plt.show()



