import torch
import torch.nn as nn


input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 3, 3]    # hello中各个字符的下标
y_data = [3, 1, 2, 3, 2]    # ohlol中各个字符的下标

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data] # (seqLen, inputSize)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size) # (seqLen, batchSize, inputSize)
labels = torch.LongTensor(y_data).view(-1, 1)   # torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
print(inputs.shape, labels.shape)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs, hidden)   # (batch_size, hidden_size)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


net = Model(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

epochs = 15

for epoch in range(epochs):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string:', end='')
    # inputs的维度是 [seqLen, batchSize, inputSize], input的维度是 [batchSize, inputSize]
    # labels的维度是 [seqLen, 1], label的维度是[1]
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 注意交叉熵在计算loss的时候维度关系，这里的hidden是([1, 4]), label是 ([1])
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))


