import torch
from torch import Tensor


def convert_to_torch_tensor(inputs, device=None):
    for i in inputs.keys():
        if isinstance(inputs[i], Tensor):
            inputs[i] = inputs[i].to(device)
        inputs[i] = torch.tensor(inputs[i], device=device)

