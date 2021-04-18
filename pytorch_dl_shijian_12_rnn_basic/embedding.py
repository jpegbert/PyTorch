import torch as t
from torch import nn as nn

"""
词向量
"""
# Embedding(num_embeddings,embedding_dim)
embedding = nn.Embedding(10, 2)  # 10个词，每个词2维
input = t.arange(0, 6).view(3, 2).long()  # 三个句子，每个句子有两个词
input = t.tensor([[9, 3], [1, 7], [3, 6]])
print("input", input)
print("embedding", embedding.weight)

input = t.autograd.Variable(input)
output = embedding(input)
print("output: ", output)
print(output.size())
print(embedding.weight.size())
