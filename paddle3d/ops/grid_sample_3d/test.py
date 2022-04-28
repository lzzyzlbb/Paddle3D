import numpy as np

import paddle
import torch
import grid_sample_3d
import kornia
import pickle

#data = np.random.rand(1, 64, 280, 376, 25).astype(np.float32)
#data = np.random.rand(1, 2, 3, 4, 5).astype(np.float32)
with open('/workspace/paddle/CaDDN/test_sample.pkl', 'rb') as f:
    dicts = pickle.load(f, encoding='latin1')
data = dicts['x']
data_t = torch.from_numpy(data)
data_t.requires_grad_()
data_p = paddle.to_tensor(data)
data_p.stop_gradient = False
grid_t = kornia.create_meshgrid3d(280, 376, 25)
grid_t = kornia.create_meshgrid3d(3, 4, 5)
grid_t = torch.from_numpy(dicts['grid'])
print(grid_t.shape)
grid_p = paddle.to_tensor(grid_t.cpu().numpy())
grid_t.requires_grad_()

grid_p.stop_gradient = False
output_t = torch.nn.functional.grid_sample(input=data_t, grid=grid_t, mode='bilinear', padding_mode='zeros', align_corners=False)
print(output_t)

output_p = grid_sample_3d.grid_sample_3d(x=data_p, grid=grid_p, mode='bilinear', padding_mode='zeros', align_corners=False)
print(output_p)
print(np.testing.assert_allclose(output_t.detach().cpu().numpy(), output_p.numpy()))
exit()
# print(output_t)

# print(output_p)

output_p.sum().backward()
output_t.sum().backward()

print(np.testing.assert_allclose(grid_t.grad, grid_p.grad))
print(data_p.grad.numpy())
print("=========")
print(data_t.grad.numpy())
print("=========output_t")
print(grid_p.grad.numpy())
print("=========")
print(grid_t.grad.numpy())
