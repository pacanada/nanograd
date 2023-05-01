from exploration import Value
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