import torch


def convert_to_torch_tensor(inputs, device=None):
    for i in inputs.keys():
        inputs[i] = torch.tensor(inputs[i], device=device)


def smart_from_numpy(ndarray, device=torch.device("cpu")):
    return torch.from_numpy(ndarray).pin_memory().to(device)
