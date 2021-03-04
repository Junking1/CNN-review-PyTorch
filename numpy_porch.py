import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)

tensor2array = torch_data.numpy()
print(
    '\nnumpydata', np_data,
    '\ntorchdata', torch_data,
    '\ntensor2array', tensor2array,
)


# abs

data = [[3, 4], [1, 2]]
tensor = torch.FloatTensor(data)  # 32bit

data = np.array(data)

print(
    '\nnumpydata', np.matmul(data,data),
    '\ntorchdata', torch.mm(tensor,tensor),
)