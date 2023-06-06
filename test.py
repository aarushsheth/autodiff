from engine import Value
import torch

def test_value_class():
    # Test 1: Simple addition
    a = Value(2.0)
    b = Value(3.0)
    f = a + b
    f.backward()
    assert f.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0

    # Test 2: Simple multiplication
    a = Value(2.0)
    b = Value(3.0)
    f = a * b
    f.backward()
    assert f.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0

    # Test 3: More complex function
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    f = a * b + c
    f.backward()
    assert f.data == 10.0
    assert a.grad == 3.0
    assert b.grad == 2.0
    assert c.grad == 1.0

    # Test 4: Test power function
    a = Value(2.0)
    f = a ** 2
    f.backward()
    assert f.data == 4.0
    assert a.grad == 4.0


    # Test 5: Test complex functions with ReLU and more operations
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    print(f.data, a.grad, b.grad)  # You need to check the results manually.

test_value_class()
