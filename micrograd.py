import math


class Value:
    def __init__(self,data,_childern=(),_op="",label = ""):
        self.data      = data
        self.grad      = 0.0
        self.label     = label
        self._op       = _op
        self._backward = lambda:None # no effects on leaf nodes, 
        self._prev     = set(_childern)
        
        
    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other) 
        out   = Value(self.data + other.data,(self,other),"+",)
        def _backward():
            self.grad  += 1.0 * out.grad    # d(self + other) / d (self) = 1
            other.grad += 1.0 * out.grad    # d(self + other) / d (other) = 1
            
        out._backward = _backward
        return out
     
    def __radd__(self,other):                    # int * Value(),eg: 1* Value(2)
        return self + other
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out   = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad  += other.data * out.grad   # d(self * other) / d(self) = other
            other.grad += self.data * out.grad    # d(self * other) / d(self) = self
        out._backward = _backward
        return out
    
    def __rmul__(self,other):                     # int * Value(), eg: 1* Value(2)
        return self * other  
    
    def __truediv__(self,other):
        return self * other ** -1
    
    def __neg__(self):
        return self * -1
        
    def __sub__(self,other):
        return self + (-other) 
    
    def __pow__(self,other):                                          # power opeartion
        assert isinstance(other,(int,float)),"Exponent must be an integer or float"
        out = Value(self.data**other, (self,), f'**{other}')          
        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad # power rule is given in the above,
        out._backward  = _backward
        return out 
    
    def exp(self): 
        x   = self.data
        out = Value(math.exp(x),(self,),label="exp") 
        def _backward():                             # differential of e^x is e^x
            self.grad += out.data * out.grad         
        out._backward = _backward
        return out 
    
    def tanh(self):
        x   = self.data
        t   = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t,(self,),_op="tanh")    
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward  = _backward
        return out
            
    def backward(self):
        topological = []
        visited = set()
        def create_topological(output_node):
            if output_node not in visited:
                visited.add(output_node) 
                for child in output_node._prev:
                    create_topological(child)
                topological.append(output_node)
        create_topological(self) 
        
        self.grad = 1.0
        for node in reversed(topological):
            node._backward()



if __name__ == "__main__":
    #inputs
    x1 = Value(2.0,label = "x1")
    x2 = Value(0.0,label = "x2")

    #weights
    w1 = Value(-3.0,label = "w1")  
    w2 = Value(1.0,label = "w2")

    #bias
    b  = Value(6.8813735870195432,label = "b") 

    #dot product
    x1w1     = x1 * w1;     x1w1.label = "x1*w1"
    x2w2     = x2 * w2;     x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2" # sum of product of weights and input ()

    # cell body 
    n = x1w1x2w2 + b; n.label = "n"

    # activation function
    o = n.tanh(); o.label = "o"
    o.backward()

    print(o.grad)