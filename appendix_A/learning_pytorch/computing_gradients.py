import torch
import torch.nn.functional as F
from torch.autograd import grad

# Actual target
y = torch.tensor([1.0])

# Input
x1 = torch.tensor([1.1])

# Trainable parameters
# requires_grad=True tells pytorch to track these tensors
# because we want to calculate how the loss depends on them.
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)


# Forward pass
z = x1 * w1 + b
a = torch.sigmoid(z)

# Compare prediction (a) with target (y)
loss = F.binary_cross_entropy(a, y)


# Calculate gradient of loss with respect to w1
grad_L_w1 = grad(loss, w1, retain_graph=True)

# Calculate gradient of loss with respect to b
grad_L_b = grad(loss, b, retain_graph=True)

print("Loss:", loss)
print("Gradient w.r.t w1:", grad_L_w1)
print("Gradient w.r.t b:", grad_L_b)


# Calculate gradients for all leaf tensors that require gradients.
# Pytorch stores the results in their .grad attributes.
loss.backward()

print("w1.grad:", w1.grad)
print("b.grad:", b.grad)
