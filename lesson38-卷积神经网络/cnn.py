import torch
from torch import nn
import torch.nn.functional as F


layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
x = torch.rand(1, 1, 28, 28)
out = layer.forward(x)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
out = layer.forward(x)

layer = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
out = layer.forward(x)

out = layer(x)

print(layer.weight)
print(layer.weight.shape)
print(layer.bias)
print(layer.bias.shape)


w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=1, padding=1)
out = F.conv2d(x, w, b, stride=2, padding=2)

print(out.shape)
