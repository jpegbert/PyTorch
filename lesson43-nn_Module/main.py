import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()

        # requires_grad = True
        self.w = nn.Parameter(torch.randn(outp, inp))
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, stride=1, padding=1),
                                 nn.MaxPool2d(2, 2),
                                 Flatten(),
                                 nn.Linear(1*14*14, 10))

    def forward(self, x):
        return self.net(x)


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(BasicNet(), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device('cuda')
    net = Net()
    net.to(device)

    net.train()

    net.eval()

    # 加载
    # net.load_state_dict(torch.load('ckpt.mdl'))
    #
    # 保存
    # torch.save(net.state_dict(), 'ckpt.mdl')

    for name, t in net.named_parameters():
        print('parameters:', name, t.shape)

    for name, m in net.named_children():
        print('children:', name, m)

    for name, m in net.named_modules():
        print('modules:', name, m)

    print(list(net.parameters())[0].shape)
    print(list(net.parameters())[3].shape)
    print(list(net.named_parameters())[0])
    print(list(net.named_parameters())[1])
    print(dict(net.named_parameters()).items())


if __name__ == '__main__':
    main()
