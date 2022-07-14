import torch

x = torch.ones(2, 2, requires_grad=True)
# print(x)
y=torch.randn(2,2,requires_grad=True)
# y = x + 2
# print(y)

z = x * y
out = z.mean()


# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

print(z)
print(out)
out.backward()
print(x.grad)