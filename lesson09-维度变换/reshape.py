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
print(b.expand(-1, 32, -1, -1).shape) # torch.Size(1, 32, 1, 1)
print(b.expand(-1, 32, -1, -4).shape) # torch.Size(1, 32, 1, -4)

# repeat
print(b.shape) # torch.Size([1, 32, 1, 1])
print(b.repeat(4, 32, 1, 1).shape) # torch.Size([4, 1024, 1, 1])
print(b.repeat(4, 1, 1, 1).shape) # torch.Size([4, 32, 1, 1])
print(b.repeat(4, 1, 32, 32).shape) # torch.Size([4, 32, 32, 32])

# transpose
print(a.shape) # [4, 3, 32, 32]
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)
print(a1.shape, a2.shape) # torch.Size(4, 3, 32, 32) torch.Size(4, 3, 32, 32)
print(torch.all(torch.eq(a, a1))) # tensor(0, dtype=torch.uint8)
print(torch.all(torch.eq(a, a2))) # tensor(1, dtype=torch.uint8)

# permute
a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape) # torch.Size(4, 28, 28, 3)
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape) # torch.Size(4, 32, 28, 3)
print(b.transpose(1, 3).transpose(1, 2).shape) # torch.Size([4, 28, 32, 3])
print(b.permute(0, 2, 3, 1).shape) # torch.Size([4, 28, 32, 3])

