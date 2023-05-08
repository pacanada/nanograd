from typing import Callable
from nanograd.engine import Value

def compute_derivative(func: Callable, var_kargs: dict, var_name: str, h: float = 1e-5) -> float:
    """ Helper function to compute derivative of a function"""
    out1 = func(**var_kargs)
    var_kargs[var_name] += h 
    out2 = func(**var_kargs)
    return (out2-out1)/h


def test_grad_simple():
    a = Value(1.0, label="a")
    b = Value(2.0, label="b")
    c = Value(3.0, label="c")
    d = Value(4.0, label="d")

    e = Value(2)*a + b + c * d
    e.label="e"
    e.backward()
    # de/da = 2
    assert a.grad == 2
    assert b.grad == 1
    assert c.grad == d.data
    assert d.grad == c.data
    assert e.grad == 1

def test_two_grad():
    a = Value(2)
    b = Value(3)

    c = a + a + b
    c.backward()
    assert a.grad == 2 
    assert b.grad == 1
    assert c.grad == 1  

def test_pow():
    a = Value(2)
    b = a**Value(3)
    b.backward()
    assert a.grad == 3*2**2

def test_div():
    a = Value(2)
    b = Value(3)
    c = a/b
    c.backward()
    assert a.grad == 1/3
    # dc/db = a*b^(-2)*(-1)
    assert b.grad == -2*3**-2

def test_substract():
    a = Value(2)
    b = Value(3)
    c = a-b
    c.backward()
    assert a.grad == 1
    assert b.grad == -1

def test_relu():
    a = Value(2)
    b = a.relu()
    b.backward()
    assert a.grad == 1

    c = Value(-1)
    d = c.relu()
    d.backward()
    assert c.grad == 0


def test_more_complex_functions():
    eps = 1e-5
    f = lambda x,y,z,t: (x*y + x**2)*z + (t*z*x)

    a, b, c, d = Value(5), Value(3), Value(-1), Value(6)
    out_autograd = f(x=a, y=b, z=c, t=d)
    out_autograd.backward()

    for value, var in zip([a,b,c,d], ["x", "y", "z", "t"]):
        diff_derivative = value.grad/compute_derivative(f, {"x":5, "y":3, "z":-1, "t": 6}, var) 
        assert 1-eps < diff_derivative and 1+eps > diff_derivative

