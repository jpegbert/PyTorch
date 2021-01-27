import torch


a = torch.rand([4, 1, 28, 28])
print(a.shape) # torch.Size([4, 1, 28, 28])
res = a.view(4, 28, 28)
print(res)
# tensor([[, , , ,],
# [, , , ,].
# [, , , ,].
# [, , , ,]])
print(a.view(4, 28 * 28).shape) # torch.Size([4, 784])
print(a.view(4 * 28, 28).shape) # torch.Size([112, 28])
print(a.view(4 * 1, 28, 28).shape) # torch.Size([4, 28, 28])
b = a.view(4, 784)
b.view(4, 28, 28, 1) # logic bug


# unsqueeze
print(a.shape) # torch.Size([4, 1, 28, 28])
print(a.unsqueeze(0).shape) # torch.Size([1, 4, 1, 28, 28])
print(a.unsqueeze(-1).shape) # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(4).shape) # torch.Size([4, 1, 28, 28, 1])
print(a.unsqueeze(-4).shape) # torch.Size([4, 1, 1, 28, 28])
print(a.unsqueeze(-5).shape) # torch.Size([1, 4, 1, 28, 28])
a = torch.tensor([1.2, 2.3])
print(a.unsqueeze(-1))
# tensor([[1.2000],
#         [2.3000]])
print(a.unsqueeze(0))
# tensor([[1.2000, 2.3000]])

# example
b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape) # torch.Size([1, 32, 1, 1])

# squeeze
print(b.shape) # torch.Size([1, 32, 1, 1])
print(b.squeeze().shape) # torch.Size([32])
print(b.squeeze(0).shape) # torch.Size([32, 1, 1])
print(b.squeeze(-1).shape) # torch.Size([1, 32, 1])
print(b.squeeze(1).shape) # torch.Size([1, 32, 1, 1])
print(b.squeeze(-4).shape) # torch.Size([32, 1, 1])

# expand / expand_as
a = torch.rand(4, 32, 14, 14)
print(b.shape) # torch.Size([1, 32, 1, 1])
print(b.expand(4, 32, 14, 14).shape) # torch.Size(4, 32, 14, 14)





