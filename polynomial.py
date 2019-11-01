import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt  

w_target = np.array([0.5,3,2.4])   #定义参数
b_target = np.array([0.9]) #定义参数
f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_target[0],w_target[0],w_target[1],w_target[2]) #打印出函数的式子
print(f_des)

#画出这个函数的曲线
x_sample = np.arange(-3,3.1,0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2 + w_target[2] * x_sample ** 3
plt.plot(x_sample,y_sample,label = 'real curve')
plt.legend()
plt.show()

# 构建数据 x 和 y
# x 是一个如下矩阵[x,x^2,x^3]
# y 是函数的结果[y]
x_train = np.stack([x_sample ** i for i in range(1,4)],axis=1)
x_train = torch.from_numpy(x_train).float() # 转换成 float tensor
y_train = torch.from_numpy(y_sample).float().unsqueeze(1) # 转换成 float tensor
#print(x_train)
#print(y_train)

w = Variable(torch.randn(3,1),requires_grad=True)
b = Variable(torch.zeros(1),requires_grad=True)
print(w)
print(b)

# 将x和y转换成Variable
x_train = Variable(x_train)
y_train = Variable(y_train)

def multi_linear(x):
    return torch.mm(x,w) + b

# 计算误差
def get_loss(y_,y):
    return torch.mean((y_- y) ** 2)

# 画出更新之前的模型
y_pred = multi_linear(x_train)
plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label='fitting curve',color='r')
plt.plot(x_train.data.numpy()[:,0],y_sample,label='real curve',color='b')
plt.legend()
plt.show()

loss = get_loss(y_pred,y_train)
print(loss)
# 自动求导
loss.backward()
print(w.grad)
print(b.grad)
# 跟新参数
w.data = w.data - 0.001 * w.grad.data
b.data = b.data - 0.001 * b.grad.data
print(w)
print(b)
# 再画一次
y_pred = multi_linear(x_train)
plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label='fitting curve',color='r')
plt.plot(x_train.data.numpy()[:,0],y_sample,label='real curve',color='b')
plt.legend()
plt.show()


# 进行100次迭代
for e in range(100):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred,y_train)
    w.grad.data.zero_()
    b.grad.data.zero_()
    loss.backward()
    w.data = w.data - 0.001 * w.grad.data
    b.data = b.data - 0.001 * b.grad.data
    if (e+1) % 10 == 0:
        print('epoch {}, Loss: {:.5f}'.format(e+1,loss))

y_pred = multi_linear(x_train)
plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label='fitting curve',color='r')
plt.plot(x_train.data.numpy()[:,0],y_sample,label='real curve',color='b')
plt.legend()
plt.show()    

