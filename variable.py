import torch
from torch.autograd import Variable 
# Create Variable
x = Variable(torch.Tensor([1]),requires_grad=True)
w = Variable(torch.Tensor([2]),requires_grad=True)
b = Variable(torch.Tensor([3]),requires_grad=True)

# Build a computational graph
y = w * x + b # y = 2*1+3

# Compute gradients
y.backward()

# Print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)

x = torch.randn(3)
x = Variable(x,requires_grad=True)

y = x * 2
print(y)
y.backward(torch.FloatTensor([1,0.1,0.01]))
print(x.grad)
