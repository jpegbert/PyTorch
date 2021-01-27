import torch


a = torch.rand(4, 3, 28, 28)
# indexing
print(a[0, 0].shape) # troch.Size([28, 28])
print(a[0, 0, 2, 4]) # tensor(0.8082)

# select first/last N
print(a.shape) # torch.Size([4,3,28,28])
print(a[:2].shape) # torch.Size([2,3,28,28])
print(a[:2, :1, :, :]) # torch.Size([2, 1, 28, 28])
print(a[:2, 1:, :, :]) # torch.Size([2, 2, 28, 28])
print(a[:2, -1:, :, :]) # torch.Size([2, 1, 28, 28])

# select by steps
print(a[:, :, 0:28:2, 0:28:2].shape) # torch.Size([4, 3, 14, 14])
print(a[:, :, 0::2, 0::2].shape) # torch.Size([4, 3, 14, 14])

# select by specific index
print(a.shape) # torch.Size(4, 3, 28, 28)
print(a.index_select(2, torch.arange(28)).shape) # torch.Size([4, 3, 28, 28])
print(a.index_select(2, torch.arange(8)).shape) # torch.Size([4, 3, 8, 28])
print(a.shape) # torch.Size([4, 3, 28, 28])
print(a[...].shape) # torch.Size([4, 3, 28, 28])
print(a[0, ...].shape) # torch.Size([3, 28, 28])
print(a[:, 1, ...].shape) # torch.Size([4, 28, 28])
print(a[..., :2].shape) # torch.Size([4, 3, 28, 2])

# select by mask
x = torch.randn(3, 4)
# tensor([[1.3911, -0.7871, -1.6558, -0.2542],
#         [..., 0.5404, ..., ...],
#         [..., ..., 0.6040, 1.5771]])
mask = x.ge(0.5)
print(mask)
# tensor([[0, 0, 0, 0],
# [0, 1, 0, 0],
# [0, 0, 1, 1]], dtype=torch.uint8)
res = torch.masked_select(x, mask)
print(res)
# tensor([0.5404, 0.6040, 1.5771])
res = torch.masked_select(x, mask).shape
print(res) # torch.Size([3])

# select by flatten index
src = torch.tensor([4, 3, 5], [6, 7, 8])
res = torch.take(src, torch.tensor([0, 2, ]))
print(res)
