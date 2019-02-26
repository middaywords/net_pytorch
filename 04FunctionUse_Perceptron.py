# one layer perceptrons
import torch
import torch.nn.functional as F

x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)

o = torch.sigmoid(x@w.t())
print(o.shape)

loss = F.mse_loss(torch.ones(1, 1), o)
loss.backward()

print(w.grad)

# multi-layer perceptrons
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
o = torch.sigmoid(x@w.t())
loss = F.mse_loss(torch.ones(1, 2), o)
loss.backward()
print(w.grad)
