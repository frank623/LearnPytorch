from __future__ import print_function
import torch as tc
import numpy as np

x = tc.empty(5,3)
print(x)

x = tc.rand(5,3)
print(x)

x = tc.zeros(5,3,dtype = tc.long)
print(x)

x = tc.tensor([5.5,3])
print(x)

x = tc.zeros(5,3,dtype=tc.long)
x = x.new_ones(5,3,dtype = tc.double)
print(x)
x = tc.rand_like(x,dtype=tc.float)
print(x)
print(x.size())

y = tc.rand(5,3)
print(x + y)
print(tc.add(x,y))

result = tc.empty(5,3)
tc.add(x,y,out=result)
print(result)

y.add_(x)
print(y)

print(x[:,1])
x = tc.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())

x = tc.randn(1)
print(x)
print(x.item())

a = tc.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = tc.from_numpy(a)
np.add(a,1,out=a)
print(a)s
print(b)

if tc.cuda.is_available():
    device = tc.device("cuda")
    y = tc.ones_like(x,device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu",tc.double))