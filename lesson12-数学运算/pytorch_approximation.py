import torch


a = torch.tensor(3.14)
print(a.floor(), a.ceil(), a.trunc(), a.frac()) # 3. 4. 3. 0.1400

a = torch.tensor(3.499)
print(a.round()) # 3.



