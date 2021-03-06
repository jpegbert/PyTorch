import torch
import torch.nn.functional as F


def sigmoid_activation():
    a = torch.linspace(-100, 100, 10)
    print(torch.sigmoid(a))


def tanh_activation():
    a = torch.linspace(-1, 1, 10)
    print(torch.tanh(a))


def relu_activation():
    a = torch.linspace(-1, 1, 10)
    print(torch.relu(a))
    print(F.relu(a))


def main():
    sigmoid_activation()
    tanh_activation()
    relu_activation()


if __name__ == '__main__':
    main()
