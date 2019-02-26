import torch
import torch.nn.functional as F


# it seems that backward cannot get the same result...?
print('\n backward')
for i in range(3):
    a1 = torch.Tensor([1, 2, 3])
    a1.requires_grad_()
    p = F.softmax(a1, dim=0)
    p[i].backward(retain_graph=True)
    print(a1.grad)

