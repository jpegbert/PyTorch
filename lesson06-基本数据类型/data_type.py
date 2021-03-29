import torch
import numpy as np


a = torch.randn(2, 3)
print(a.type()) # torch.FloatTensor
print(type(a)) # torch.Tensor
print(isinstance(a, torch.FloatTensor)) # true

data = torch.randn(2, 3)
print(isinstance(data, torch.cuda.DoubleTensor)) # False
data = data.cuda()
print(isinstance(data, torch.cuda.DoubleTensor)) # True

print(torch.tensor(1.)) # tensor(1.)
print(torch.tensor(1.3)) # tensor(1.300)

a = torch.tensor(2.2)
print(a.shape) # torch.Size([])
print(len(a.shape)) # 0
print(a.size()) # torch.Size([])

print(torch.tensor([1.1])) # tensor([1.1000])
print(torch.tensor([1.1, 2.2])) # tensor([1.1000, 2.2000])
print(torch.FloatTensor(1)) # tensor([3.2239e-25])
print(torch.FloatTensor(2)) # tensor([3.2239e-25, 4.5915e-41])

data = np.ones(2)
print(data) # array([1., 1.])
print(torch.from_numpy(data)) # tensor([1., 1.], dtype=torch.float64)

a = torch.ones(2)
print(a.shape) # torch.Size([2])
a = torch.randn(2, 3)
print(a)
# tensor([[-0.4423, 0.5949, 1.1440],
#   [-2.0935, 0.2051, 1.2781]])
print(a.shape) # torch.Size([2, 3])
print(a.size(0)) # 2
print(a.size(1)) # 3
print(a.shape[1]) # 3

a = torch.rand(1, 2, 3)
print(a)
# tensor([[[0.0764, 0.2590, 0.9816],
#   [0.6798, 0.1568, 0.7919]]])
print(a.shape) # torch.Size(1, 2, 3)
print(a[0])
# tensor([[0.0764, 0.2590, 0.9816],
#   [0.6798, 0.1568, 0.7919]])
print(list(a.shape)) # [1, 2, 3]

a = torch.rand(2, 3, 28, 28)
print(a.shape) # torch.Size([2, 3, 28, 28])
print(a.numel()) # 4704
print(a.dim()) # 4
a = torch.tensor(1)
print(a.dim()) # 0

