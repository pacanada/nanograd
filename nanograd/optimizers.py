from nanograd.engine import Value
class SGD:
    def __init__(self, params: list[Value], lr: float=0.1):
        self.params = params
        self.lr = lr
    def step(self):
        for p in self.params:
            p.data -= self.lr*p.grad