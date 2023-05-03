import random
import numpy as np
from nn import MLP

from graphviz import Digraph
import matplotlib.pyplot as plt
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, filename, format='png', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(filename=filename, format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
np.random.seed(1337)
random.seed(1337)
N = 1000
N_ITER = 500
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
mlp = MLP(nin=SEQ_SIZE, nouts=[5,5,5,SEQ_SIZE])
print("Number of params ", len(mlp.parameters()))
# optim = SGD(mlp.parameters(), lr)

for i in range(N_ITER):
    #idx = np.random.randint(N, size=SEQ_SIZE)
    idx = np.random.randint(N, size=BATCH_SIZE)
    inputs = [[i] for i in x[idx]]

    out = list(map(mlp, inputs))
    out = [i for j in out for i in j]
    #out=mlp(x[idx])
    
    data_loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in mlp.parameters()))
    loss = data_loss+reg_loss
    #loss = sum([(t-i)**2 for t, i in zip(y[idx], out)])/len(out)
    #loss = mse(out, y[idx])
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

plt.plot(x_test, [i[0].data for i in list(map(mlp,[[x0] for x0 in x_test]))], label="pred")
plt.plot(x_test, y_test, label="target")
plt.legend()
plt.show()

plt.plot(x[0:SEQ_SIZE], y[0:SEQ_SIZE], label="true")

print([out.data for out in mlp(x[0:SEQ_SIZE])])
# plt.plot(x[0:SEQ_SIZE], [out.data for out in mlp(x[0:SEQ_SIZE])], label="pred")
# plt.legend()
# plt.show()

