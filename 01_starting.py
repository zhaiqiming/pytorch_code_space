from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

# 
x = torch.rand(5, 3)
print(x)

# 构造一个矩阵全为 0，而且数据类型是 long.
x = torch.zeros(5, 3, dtype=torch.long)
print(x)


# 构造一个张量，直接使用数据：
x = torch.tensor([5.5, 3])
print(x)

# 创建一个 tensor 基于已经存在的 tensor。
x = x.new_ones(5, 3, dtype=torch.double)      
# new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    
print(x)                                      

# 获取它的维度信息:
print(x.size())

y = torch.rand(5, 3)
print(x + y)

# adds x to y
y.add_(x)
print(y)