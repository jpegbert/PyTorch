import torch
from torch import nn
import torch.nn.functional as F


w = torch.rand(16, 3, 5, 5)
b = torch.rand(16)
x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, w, b, stride=1, padding=1)
out = F.conv2d(x, w, b, stride=2, padding=2)

print(out.shape)

x = out
# pooling
layer = nn.MaxPool2d(2, stride=2)
out = layer(x)
print(out.shape)
out = F.avg_pool2d(x, 2, stride=2)
print(out.shape)


x = out
out = F.interpolate(x, scale_factor=2, mode="nearest")
print(out.shape)
out = F.interpolate(x, scale_factor=3, mode="nearest")
print(out.shape)


# ReLU
layer = nn.ReLU(inplace=True)
out = layer(x)
print(out.shape)
out = F.relu(x)
print(out.shape)
