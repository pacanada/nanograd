import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
np.random.seed(1337)
random.seed(1337)
N = 1000
N_ITER = 100
SEQ_SIZE = 1
BATCH_SIZE = 100
PLOT = False
lr = 0.0001
loss_l = []
f = lambda x: x**2
x = np.linspace(0,10, N)
x_test = np.linspace(10,20, N)
y = f(x) #np.sin(x)
y_test = f(x_test)


#mlp = MLP((SEQ_SIZE,1,SEQ_SIZE))
#mlp = MLP(nin=SEQ_SIZE, nouts=[5,5,5,SEQ_SIZE])
mlp=MLPRegressor(hidden_layer_sizes=(5,5,5), batch_size=BATCH_SIZE, warm_start=True, solver="sgd")
#print("Number of params ", len(mlp.parameters()))
# optim = SGD(mlp.parameters(), lr)

for i in range(N_ITER):
    #idx = np.random.randint(N, size=SEQ_SIZE)
    #idx = np.random.randint(N, size=BATCH_SIZE)
    #inputs = [[i] for i in x[idx]]

    #out = list(map(mlp, inputs))
    #out = [i for j in out for i in j]
    #out=mlp(x[idx])
    
    # data_loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    # alpha = 1e-4
    # reg_loss = alpha * sum((p*p for p in mlp.parameters()))
    # loss = data_loss+reg_loss
    # #loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    # #loss = mse(out, y[idx])
    # mlp.zero_grad()
    # loss.backward()
    #lr = 1.0 - 0.9*i/100
    # for p in mlp.parameters():
    #     p.data -= lr * p.grad
    #optim.step()
    mlp=mlp.fit(x.reshape(-1,1),y)

    print(mlp.loss_)
    loss_l.append(mlp.loss_)

print("For 2 we have ", mlp.predict(X=np.linspace(2,2,1).reshape(-1,1)))
plt.plot(np.log(loss_l))
plt.show()

plt.plot(x_test, mlp.predict(x_test.reshape(-1,1)), label="pred")
plt.plot(x_test, y_test, label="target")
plt.legend()
plt.show()

plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")

#print([out.data for out in mlp(x[0:SEQ_SIZE])])
# plt.plot(x[0:SEQ_SIZE], [out.data for out in mlp(x[0:SEQ_SIZE])], label="pred")
# plt.legend()
# plt.show()

