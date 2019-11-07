import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# y = 0.9 + 0.5 * x  + 3 * x * x + 2.4 * x * x * x
def make_features(x):
    #Builds features i.e. a matrix with columns [x,x^2,x^3]
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,4)],1)

W_target = torch.FloatTensor([0.5,3,2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    # Approximated function
    return x.mm(W_target) + b_target[0]

def get_batch(batch_size = 32):
    random = torch.randn(batch_size)
    randomsort,randomince = torch.sort(random)
    #random = np.arange(-3,3.1,0.1)
    #random = torch.from_numpy(random)
    x = make_features(randomsort)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x) , Variable(y)
    
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model,self).__init__()
        self.poly = nn.Linear(3,1)
    
    def forward(self,x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
#get data
batch_x,batch_y = get_batch()
print(batch_x)
print(batch_y)
while True:
    #forward pass
    output = model(batch_x)
    loss = criterion(output,batch_y)
    #Reset gradients
    optimizer.zero_grad()
    #Backward pass
    loss.backward()
    #update parameters
    optimizer.step()
    epoch += 1
    if loss < 1e-3:
        break

print('Loss:{:.6f} after {} batches'.format(loss,epoch))
#print('==> Learned function: y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(model.parameters))
#print('==> Actual function:  y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(b_target[0],W_target[0],W_target[1],W_target[2]))
plt.plot(batch_x.cpu().numpy()[:,0],batch_y.cpu().numpy(),'bo',label='raal curve',color='b')
plt.plot(batch_x.cpu().numpy()[:,0],output.cpu().data.numpy(),label='Fitting curve',color='r')
plt.legend()
plt.show()

