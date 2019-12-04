import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
import time

torch.manual_seed(2019)
#从data.txt中读入数据点
with open('data.txt','r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]

#标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max,i[1]/x1_max,i[2]) for i in data]

x0 = list(filter(lambda x:x[-1] == 0.0 , data)) #选择第一类的点
x1 = list(filter(lambda x:x[-1] == 1.0 , data)) #选择第二类的点

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

plt.plot(plot_x0,plot_y0,'ro',label='x_0')
plt.plot(plot_x1,plot_y1,'bo',label='x_1')
plt.legend(loc='best')
plt.show()

np_data = np.array(data,dtype='float32') #转换为numpy array
x_data = torch.from_numpy(np_data[:,0:2]) #转换为tensor ，大小是[100，2]
y_data = torch.from_numpy(np_data[:,-1]).unsqueeze(1) #转换为tensor，大小是[100,1]

#定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#画出sigmoid的图像
plot_x = np.arange(-10,10.01,0.01)
plot_y = sigmoid(plot_x)

plt.plot(plot_x,plot_y,'r')
plt.show()

x_data = Variable(x_data)
y_data = Variable(y_data)

#定义logistic回归模型
w = Variable(torch.randn(2,1),requires_grad=True)
b = Variable(torch.zeros(1),requires_grad = True)

def logistic_regression(x):
    return torch.sigmoid(torch.mm(x,w) + b)

w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_x = torch.from_numpy(plot_x)
plot_x = Variable(plot_x)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x,plot_y,'g',label='cuttin line')
plt.plot(plot_x0,plot_y0,'ro',label='x_0')
plt.plot(plot_x1,plot_y1,'bo',label='x_1')
plt.legend(loc='best')
plt.show()

def binary_loss(y_pred,y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 -y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits

y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred,y_data)
print(loss)

loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data

y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred,y_data)
print(loss)

w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))

def logistic_regression(x):
    return torch.sigmoid(torch.mm(x,w) + b)

optimizer = torch.optim.SGD([w,b],lr=1.)

start = time.time()
for e in range(1000):
    #前向传播
    y_pred = logistic_regression(x_data)
    loss = binary_loss(y_pred,y_data) #计算loss
    #反向传播
    optimizer.zero_grad() #使用优化器将梯度归0
    loss.backward()
    optimizer.step() #使用优化器更新参数
    #计算正确率
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum() / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch:{},Loss:{:.5f},Acc:{:.5f}'.format(e+1,loss,acc))
during = time.time() - start
print()
print('During Time:{:.3f}s'.format(during))

w0 = w[0].data[0]
w1 = w[1].data[0]
b0 = b.data[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_x = torch.from_numpy(plot_x)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
plt.show()