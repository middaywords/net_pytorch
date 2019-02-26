import torch
import torch.nn.functional as F

x = torch.randn(1, 784)
w = torch.randn(10, 784)

# y = wx + 0
logits = x@w.t()
# get cross entropy
# approach 1:
#   manual compute
pred        = F.softmax(logits, dim = 1)
pred_log    = torch.log(pred)
print(F.nll_loss(pred_log, torch.tensor([3])))

# approach 2:
#   API: cross_entropy includes softmax and log operations.
print(F.cross_entropy(logits, torch.tensor([3])))