import torch

x = torch.tensor([10.0], requires_grad=True)
lr = 0.1

for epoch in range(50):
    loss = x**2 - 4 * x + 4

    loss.backward()

    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()
