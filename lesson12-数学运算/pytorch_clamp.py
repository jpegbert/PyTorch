import torch


grad = torch.rand(2, 3) * 15
print(grad.max())
print(grad.median())
# 对于小于10的换成最小值10
print(grad.clamp(10))

