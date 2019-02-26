import torch
import torch.nn.functional as F

print('torch version:', torch.__version__)
print('gpu', torch.cuda.is_available())

# broadcast 自动根据需要扩展tensor
# two feature: expand and without copy data
featureMap = torch.randn(4, 32, 14, 14)
# align from dims in the last 后面的维度定义为小维度
# 感觉和numpy的不同维度的数组相加比较像，从小维度开始对齐。
# 具体看slides
bias = torch.randn(2, 32, 1, 1)
# 这个就加不了，因为第一维不是1  uncomment to run
# featureMap2 = featureMap + bias

# merge & split
# cat:
#   from left to right, dim (0-> n-1)
a = torch.randn(5, 1, 8)
b = torch.randn(5, 32, 8)
c = torch.cat([a, b], dim=1)
print(c.shape)

# stack:
#   create new dim
a1 = a2 = torch.randn(4, 3, 16, 32)
#   change dim to see results, a1 & a2 must be same size.
c = torch.stack([a1, a2], dim=2)
print(c.shape)

# split:
#   split by length
c = torch.randn(3, 32, 8)
aa, bb = c.split([1, 2], dim=0)
print('aa', aa.shape)
print('bb', bb.shape)
#   by length!
c = torch.randn(6, 32, 8)
aa, bb, cc = c.split(2, dim=0)
print('aa', aa.shape)
print('bb', bb.shape)

# chunk
#   split by number
c = torch.randn(6, 32, 8)
aa1, bb1 = c.chunk(2, dim=0)
print(aa1.shape, bb1.shape)

# math operations
a = torch.rand(3, 4)
b = torch.rand(4)
print('a+b:', a + b)
print('.add', torch.add(a, b))
print(torch.eq(a - b, torch.sub(a, b)))

print(torch.all(torch.eq(a - b, torch.sub(a, b))))
# a*b, a/b are omitted here.
# all(): Returns True if all elements in the tensor
#    are non-zero, False otherwise.

a = torch.ones(2, 2) * 3
b = a * 3
# '*' element-wise, '@' or 'matmal()' for matrix
print(a)
print(b)
print(torch.matmul(a, b))
print(a @ b)

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print(torch.matmul(a, b).shape)

a = torch.full([2, 2], 3)
print(a.pow(2), a ** 2, a ** 0.5)

# e^x
a = torch.exp(torch.ones(2, 2))
print(a)

# approximation
a = torch.tensor(3.14)
print(a.floor(), a.ceil(), a.trunc(), a.frac(),
      torch.round(torch.tensor(3.49)), torch.round(torch.tensor(3.5)))

# clamp:
#   those smaller than 10 will be set 10
#   for instance, gradient clipping.
grad = torch.rand(2, 3) * 15
print(grad.max(), grad.median(), '\n', grad.clamp(10)
      , '\n', grad.clamp(0, 10))

# statistics
# norm: 求范数，
a = torch.full([8], 2)
b = a.view(2, 4)
c = a.view(2, 2, 2)

print(a.norm(1), b.norm(1), c.norm(1))
# mean, sum, min, max, prod,
# argmin, argmax, (parameters: dim, keepdim=True)
# topk

# advanced operations:
# where
#     params: condition: shape same as a,b
# gather, gather info by index.

# Activation funtion
sig = torch.sigmoid(a)
print(sig)

rlu = torch.relu(a)
rlue = F.relu(a)
print(rlu, rlue)

# ----------------------------------------------------------
# Loss function

x = torch.ones(1)
# set it as tensor require gradient.
w = torch.full([1], 2, requires_grad=True)
mseloss = F.mse_loss(torch.ones(1), x*w)
print('loss；', mseloss)
# but we cannot use autograd to get gradient.

# _ means it needs to implex, it will update w.
w.requires_grad_()
# but still cannot function when run autograd, we need to update the flowgraph.
# only can get before we build the graph.

# we need to set when initializing w.
print(torch.autograd.grad(mseloss, [w]))
print(' ')

x = torch.ones(1)
# set it as tensor require gradient.
w = torch.full([1], 2, requires_grad=True)
mseloss = F.mse_loss(torch.ones(1), x*w)
# it will compute grads for all tensors that requires gard.
mseloss.backward()
print(w.grad)


# it seems that backward cannot get the same result...?
print('\n backward')
a1 = torch.Tensor([1, 2, 3])
a1.requires_grad_()
p = F.softmax(a1, dim=0)
for i in range(3):
    p[i].backward(retain_graph=True)
    print(a1.grad)

print('\n autograd')
a2 = torch.Tensor([1, 2, 3])
a2.requires_grad_()
p = F.softmax(a2, dim=0)
for i in range(3):
    print(torch.autograd.grad(p[i], [a2], retain_graph=True))







