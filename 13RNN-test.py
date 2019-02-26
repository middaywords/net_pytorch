import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
# how to get some info about rnn parameters
print(rnn.parameters)
print(rnn._parameters.keys())
print('\n rnn weight vectors:')
print(rnn.weight_ih_l0.shape)

print()
