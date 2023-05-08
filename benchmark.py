import time
import numpy as np
import random
from dataclasses import dataclass
import torch.nn as nn
import torch
from nanograd.mlp import MLP
from nanograd.optimizers import SGD
from nanograd.engine import Value
import matplotlib.pyplot as plt

np.random.seed(1337)
random.seed(1337)

@dataclass
class Config:
    N = 50
    N_ITER = 5000
    LOG_EVERY = 10
    SEQ_SIZE = 1
    BATCH_SIZE = 10
    PLOT = False
    LR = 0.0001

class MLPTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,10)
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,1)
    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x

def mse(output: list[Value], target: list[Value])-> Value:
    if len(output)!=len(target):
        raise Exception(f"Output size ({len(output)}) and target size ({len(target)}) does match!")
    # good old mse
    loss = sum([(t-i)**2 for t, i in zip(target, output)])/len(output)
    return loss

def get_data(config:Config):
    """Simple x^2 transformation.
    We dont use test set since we are just benchmarking if the mlp can 
    adjust to the curve and we are not interested on the performance
    """
    f = lambda x: x**2
    #f = lambda x: np.sin(x)
    x = np.linspace(0,10, config.N)
    #x = np.linspace(0,10*np.pi, config.N)
    y = f(x)
    return x, y    


def main(config: Config):
    loss_l = []
    loss_l_torch = []
    time_elapsed = 0
    time_elapsed_torch = 0
    mlp = MLP((config.SEQ_SIZE,10,10,config.SEQ_SIZE))
    mlp_torch = MLPTorch()

    optim = SGD(mlp.parameters(), config.LR)
    optim_torch = torch.optim.SGD(mlp_torch.parameters(), config.LR)

    loss_f = mse
    loss_f_torch = torch.nn.MSELoss()

    x, y = get_data(config)

    for i in range(config.N_ITER):
        # Same idx for both models
        idx = np.random.randint(config.N, size=config.BATCH_SIZE)
        # Nanograd mlp
        t0 = time.time()
        inputs = [[i] for i in x[idx]]
        out = list(map(mlp, inputs))
        out = [i for j in out for i in j]
        loss = loss_f(out, y[idx])
        mlp.zero_grad()
        loss.backward()
        optim.step()
        t1 = time.time()

        #Torch mlp
        t0_torch = time.time()
        out = mlp_torch(torch.tensor(x[idx], dtype=torch.float32).view(-1,1))
        loss_torch = loss_f_torch(out, torch.tensor(y[idx], dtype=torch.float32).view(-1,1))
        mlp_torch.zero_grad()
        loss_torch.backward()
        optim_torch.step()
        t1_torch = time.time()

        time_elapsed +=(t1-t0)
        time_elapsed_torch +=(t1_torch-t0_torch)
        if i % config.LOG_EVERY==0:
            print(f"{i}: Nanograd mlp loss: {loss.data:.2f}. Pytorch mlp loss: {loss_torch.detach().item():.2f}")
            loss_l.append(loss.data)
            loss_l_torch.append(loss_torch.detach().item())

    # Time
    print(f"Nanograd training: {time_elapsed:.2f}s, Torch training: {time_elapsed_torch:.2f}s")
    # Losses
    plt.plot(np.log(loss_l), label="Nanograd loss")
    plt.plot(np.log(loss_l_torch), label="Pytorch loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("log (loss)")
    plt.savefig("loss.png")
    plt.show()
    # Predictions
    plt.plot(x, y, "k--", label="target")
    plt.plot(x, [i[0].data for i in list(map(mlp,[[x0] for x0 in x]))], label="nanograd")
    plt.plot(x, mlp_torch(torch.tensor(x, dtype=torch.float32).view(-1,1)).detach().numpy(), label="pytorch")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y=x^2")
    plt.savefig("x2_preds.png")
    plt.show()

if __name__ == "__main__":
    config = Config()
    main(config)