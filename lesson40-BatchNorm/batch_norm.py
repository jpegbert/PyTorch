import torch
from torch import nn


x = torch.randn(100, 16) + 0.5
print(x)
layer = torch.nn.BatchNorm1d(16)
print(layer.running_mean, layer.running_var)
out = layer(x)
print(layer.running_mean, layer.running_var)

for i in range(100):
    out = layer(x)
print(layer.running_mean, layer.running_var)


layer = nn.BatchNorm2d(16)
out = layer(x)
print(layer.weight)
print(layer.weight.shape)
print(layer.bias)
print(layer.bias.shape)


print(vars(layer))
print(layer.eval())


