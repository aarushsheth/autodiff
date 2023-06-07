# Value class starter code, with many functions taken out
from math import exp, log

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def create_value(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        # Create a new Value object with the sum of the data from self and other
        other = self.create_value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Compute gradients for self and other based on the gradient of the output
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        # Create a new Value object with the product of the data from self and other
        other = self.create_value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Compute gradients for self and other based on the gradient of the output
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += exp(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(log(self.data), (self,), 'log')
    
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        # Create a new Value object with the power of self.data raised to the other value
        out = Value(self.data**other, (self,), f'**{other}')
        
        def _backward():
            # Compute gradient for self based on the gradient of the output
            self.grad += (other * self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out

    def relu(self):
        # Apply the ReLU activation function to self.data
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            # Compute gradient for self based on the gradient of the output
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        # Perform backpropagation to compute gradients for all values in the computation graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        
        for v in reversed(topo):
            v._backward()

    def __neg__(self): 
        # Negate the value of self
        return self * -1

    def __radd__(self, other): 
        # Add self to the other value
        return self + other

    def __sub__(self, other): 
        # Subtract other value from self
        return self + (-other)

    def __rsub__(self, other): 
        # Subtract self from the other value
        return other + (-self)

    def __rmul__(self, other): 
        # Multiply self with the other value
        return self * other

    def __truediv__(self, other): 
        # Divide self by the other value
        return self * other**-1

    def __rtruediv__(self, other): 
        # Divide the other value by self
        return other * self**-1
    

    def __repr__(self):
        # Return a string representation of the Value object
        return f"Value(data={self.data}, grad={self.grad})"
