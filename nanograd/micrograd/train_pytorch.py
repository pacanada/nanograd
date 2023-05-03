import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
N = 1000
N_ITER = 100000
SEQ_SIZE = 1
BATCH_SIZE = 30
PLOT = False
lr = 0.001
loss_l = []
f = lambda x: torch.sin(x) #x**2
x = torch.linspace(0,10, N).view(N,1)
x_test = torch.linspace(10,20, N).view(N,1)
y = f(x) #np.sin(x)
y_test = f(x_test)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,10)
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,1)
        # self.l4 = nn.Linear(5,1)
    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)#.relu()
        # x = self.l4(x).relu()
        return x

#mlp = MLP((SEQ_SIZE,1,SEQ_SIZE))
mlp = MLP()
#print("Number of params ", len(mlp.parameters()))
optim = torch.optim.SGD(mlp.parameters(), lr)
#optim = torch.optim.Adam(mlp.parameters(), lr)
loss_f = torch.nn.MSELoss()

for i in range(N_ITER):
    #idx = np.random.randint(N, size=SEQ_SIZE)
    idx = np.random.randint(N, size=(BATCH_SIZE))
    out = mlp(x[idx])
    loss = loss_f(out, y[idx])
    #inputs = [[i] for i in x[idx]]

    #out = list(map(mlp, inputs))
    # out = [i for j in out for i in j]
    # #out=mlp(x[idx])
    
    # data_loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    # alpha = 1e-4
    # reg_loss = alpha * sum((p*p for p in mlp.parameters()))
    # loss = data_loss+reg_loss
    #loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    #loss = mse(out, y[idx])
    mlp.zero_grad()

    loss.backward()
    #lr = 1.0 - 0.9*i/100
    # for p in mlp.parameters():
    #     p.data -= lr * p.grad
    optim.step()
    if i%100==0:
        print(loss)
        loss_l.append(loss.detach().numpy())

#print("For 2 we have ", mlp([2]))
plt.plot(np.log(loss_l))
plt.show()

plt.plot(x_test, mlp(x_test.view(-1,1)).detach().numpy(), label="pred")
plt.plot(x_test, y_test, label="target")
plt.legend()
plt.show()

plt.plot(x, mlp(x.view(-1,1)).detach().numpy(), label="pred")
plt.plot(x, y, label="target")
plt.legend()
plt.show()

#plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")

#print([out.data for out in mlp(x[0:SEQ_SIZE])])