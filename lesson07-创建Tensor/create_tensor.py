import torch
import numpy as np


"""
创建Tensor
"""

# 从numpy数组中创建Tesnor
a = np.array([2, 3.3])
a = torch.from_numpy(a)
print(a) # tensor([2.0000, 3.3000], dtype=torch.float64)
a = np.ones([2, 3])
a = torch.from_numpy(a)
print(a) # tensor([[1., 1., 1.], [1., 1., 1.]], dtype=torch.float64)


# 从list创建Tensor
a = torch.tensor([2., 3.2])
print(a) # tensor([2.0000, 3.2000])
a = torch.FloatTensor([2., 3.2])
print(a) # tensor([2.0000, 3.2000])
a = torch.tensor([[2., 3.2], [1., 22.3]])
print(a) # tensor([[2.0000, 3.2000], [1.0000, 22.3000]])


# uninitialized 创建Tensor
a = torch.empty(1)
print(a) # tensor([0.])
a = torch.Tensor(2, 3)
print(a)
# tensor([[3.1921e+27, 0.0000e+00, -1.0163e+11], [7.1186e-43, 0.0000e+00, -0.0000e+00]])
a = torch.IntTensor(2, 3)
a = torch.FloatTensor(2, 3)


# 设置默认值
a = torch.tensor([1.2, 3]).type()
print(a.type()) # torch.FloatTensor
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 3]).type()) # torch.DoubleTensor


# rand/rand_like, randint
print(torch.rand(3, 3))
# tensor([[0.1489, 0.3093, 0.0103], [0.7305, 0.6398, 0.1361], [0.0675, 0.8197, 0.0676]])
a = torch.rand(3, 3)
b = torch.rand_like(a)
print(b)
# tensor([[0.1823, 0.2776, 0.3376], [0.2285, 0.7772, 0.9575], [0.6914, 0.4166, 0.2171]])
torch.randint(1, 10)

# randn
a = torch.randn(3, 3)
print(a)
# tensor([[-0.7416, -1.7052, -0.1960], [0.9920, 0.4750, 0.7747], [-0.3542, 0.3421, 0.5126]])
torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))


# full
a = torch.full([2, 3], 7)
print(a)
# tensor([[7., 7., 7.], [7., 7., 7.]])
a = torch.full([], 7)
print(a)
# tensor(7.)
a = torch.full([1], 7)
print(a)
# tensor([7.])


# arange/range
a = torch.arange(0, 10)
print(a)
# tensor([0,1,2,3,4,5,6,7,8,9])
a = torch.arange(0, 10, 2)
print(a)
# tensor([0, 2, 4, 6, 8])
a = torch.range(0, 10)
print(a)
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])


# linspace/logspace
a = torch.linspace(0, 10, steps=4)
print(a) # tensor([0.0000, 3.3333, 6.6667, 10.0000])
a = torch.linspace(0, 10, steps=10)
print(a) # tensor([0.0000, 1.1111, 2.2222, 3.3333, 4.4444, 5.5556, 6.6667, 7.7778, 8.8889, 10.0000])
a = torch.linspace(0, 10, steps=11)
print(a) # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
a = torch.logspace(0, -1, steps=10)
print(a) # tensor
a = torch.logspace(0, 1, steps=10)


# ones/zeros/eye
a = torch.ones(3, 3)
print(a)
# tensor([[1., 1. , 1.], [1., 1. , 1.], [1., 1. , 1.]])
a = torch.zeros(3, 3)
print(a)
# tensor([[0., 0. , 0.], [0., 0. , 0.], [0., 0. , 0.]])
a = torch.eye(3, 4)
print(a)
# tensor([[1., 0. , 0., 0.],
# [0., 1. , 0., 0.],
# [0., 0. , 1., 0.]])
a = torch.zeros(3, 3)
b = torch.ones_like(a)
print(b)
# tensor([[1., 1. , 1.], [1., 1. , 1.], [1., 1. , 1.]])


# randperm
a = torch.rand(2, 3)
b = torch.rand(2, 2)
idx = torch.randperm(2)
print(idx) # tensor([1, 0])
print(idx) # tensor([0, 1])
print(a[idx])
# tensor([[0.4283, 0.4819, 0.6252], [0.9189, 0.7713, 0.9449]])
print(b[idx])
# tensor([[0.2237, 0.6649], [0.1008, 0.7560]])

