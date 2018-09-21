import torch


def convert_to_torch_tensor(inputs, device=None):
    for i in inputs.keys():
        inputs[i] = torch.tensor(inputs[i], device=device)