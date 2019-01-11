import torch
from torch import Tensor
import torch.nn.functional as F


def convert_to_torch_tensor(inputs, device=None):
    for i in inputs.keys():
        if isinstance(inputs[i], Tensor):
            inputs[i] = inputs[i].to(device)
        inputs[i] = torch.tensor(inputs[i], device=device)


def to_cuda(inputs):
    for i in inputs.keys():
        if isinstance(inputs[i], Tensor):
            inputs[i] = inputs[i].cuda()


def pad_and_stack_1d(tensors, to=None, constant=0):
    if to is None:
        to = max(i.size(0) for i in tensors)
    shape = (len(tensors), to) + tensors[0].shape[1:]
    result = torch.full(shape, constant, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        result[idx, :tensor.shape[0]] = tensor
    return result


def pad_and_stack_2d(tensors, to_1=None, to_2=None, constant=0):
    if to_1 is None:
        to_1 = max(i.size(0) for i in tensors)
    if to_2 is None:
        to_2 = max(i.size(1) for i in tensors)
    shape = (len(tensors), to_1, to_2) + tensors[0].shape[2:]
    result = torch.full(shape, constant, dtype=tensors[0].dtype)
    for idx, tensor in enumerate(tensors):
        result[idx, :tensor.shape[0], :tensor.shape[1]] = tensor
    return result


def broadcast_gather(input, dim, index, out=None):
    if len(index.shape) < len(input.shape):
        old_shape = index.shape
        index = index.view(index.shape + (1,) * (len(input.shape) - len(index.shape)))
        expand_params = (-1,) * len(old_shape) + input.shape[len(old_shape):]
        index = index.expand(*expand_params)
    return torch.gather(input, dim, index, out=out)


def batch_index_select(input, index):
    batch_size = input.shape[0]
    assert index.shape[0] == batch_size
    input_flat = input.view(batch_size * input.shape[1], *input.shape[2:])
    start_idx = (torch.arange(batch_size, device=input.device) * input.shape[1]).unsqueeze(-1)
    index_flat = (index + start_idx).view(-1)
    result_flat = input_flat[index_flat]
    result = result_flat.view(batch_size, index.shape[1], *input.shape[2:])
    return result


def cross_entropy_nd(input, target, weight=None, size_average=None, ignore_index=-100,
                     reduce=None, reduction='mean'):
    input_flat = input.view(-1, input.shape[-1])
    target_flat = target.view(-1)
    ret = F.cross_entropy(input_flat, target_flat, weight, size_average, ignore_index,
                          reduce, reduction)
    if ret.dim() != 0:
        return ret.view(input.shape[:-1])
    return ret
