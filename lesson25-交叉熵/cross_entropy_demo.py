import torch
import torch.nn.functional as F


x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()

pred = F.softmax(logits, dim=1)
pred_log = torch.log(pred)

# Numerical Stability
res1 = F.cross_entropy(logits, torch.tensor([3]))
res2 = F.nll_loss(pred_log, torch.tensor([3]))
print(res1) # tensor(33.3334)
print(res2) # tensor(33.3334)
