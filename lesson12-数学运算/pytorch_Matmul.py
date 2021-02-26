import torch


a = torch.rand(2, 2)
b = torch.ones(2, 2)

# 第一种方法，只用于二维
torch.mm(a, b)

# 第二种方法
torch.matmul(a, b)

# 第三种方法
c = a @ b


