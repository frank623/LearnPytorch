import torch

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

