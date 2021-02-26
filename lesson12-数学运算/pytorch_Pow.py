import torch


a = torch.full([2, 2], 3)

# 方法1
a.pow(2)

# 方法2
c = a**2
d = c.sqrt() # 开方
e = c.rsqrt()

# 方法3
f = c**(0.5)

