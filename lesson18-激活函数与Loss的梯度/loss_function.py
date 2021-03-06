import torch
import torch.nn.functional as F


def mse_loss():
    x = torch.ones(1)
    w = torch.full([1], 2)

    w.requires_grad_()
    mse = F.mse_loss(torch.ones(1), x * w)
    torch.autograd.grad(mse, [w])
    # mse.backward()
    # print(w.grad)


def softmax_loss():
    a = torch.rand(3)
    a.requires_grad_()
    p = F.softmax(a, dim=0)
    torch.autograd.grad(p[1], [a], retain_graph=True)
    torch.autograd.grad(p[2], [a], retain_graph=True)


def main():
    mse_loss()
    softmax_loss()


if __name__ == '__main__':
    main()
