import torch
import torch.nn.functional as F
from torch import autograd


x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)

y1 = x * w1 + b1
y2 = y1 * w2 + b2

dy2_dy1 = autograd.grad(y2, [y1], retain_graph=True)[0]
dy1_dw1 = autograd.grad(y1, [w1], retain_graph=True)[0]
dy2_dw1 = autograd.grad(y2, [w1], retain_graph=True)[0]
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)


