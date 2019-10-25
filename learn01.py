from __future__ import print_function
import torch as tc

a = tc.empty(5,3)
print(a)

b = tc.rand(5,3)
print(b)

c = tc.zeros(5,3,dtype = tc.long)
print(c)

d = tc.tensor([5.5,3])
print(d)

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

