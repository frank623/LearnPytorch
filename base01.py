import torch
import numpy as np

#定义一个三行两列给定元素的矩阵
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is : {}'.format(a))
print('a size is {}'.format(a.size()))
#定义一个三行两列长整数的矩阵
b = torch.LongTensor([[2,3],[4,8],[7,9]])
print('b is : {}'.format(b))
#定义全是0的矩阵
c = torch.zeros((3,2))
print('c is : {}'.format(c))
#定义正态分布随机数的矩阵
d = torch.randn(3,2)
print('d is : {}'.format(d))

a[0,1] = 100
print('changed a is : {}'.format(a))
#将tensor转为numpy
numpy_b = b.numpy()
print('conver to numpy is \n {}'.format(numpy_b))
#将numpy转为tensor
e = np.array([[2,3],[4,5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()
print('change data type to float tensor：{}'.format(f_torch_e))

if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)
