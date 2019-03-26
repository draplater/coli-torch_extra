from typing import List

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
            inputs[i] = inputs[i].pin_memory().cuda()


# @torch.jit.script
def pad_and_stack_1d(tensors, to= -1, constant= 0, device=None):
    # type: (List[Tensor], int, int, Device) -> Tensor
    if to == -1:
        for i in range(len(tensors)):
            tmp = tensors[i].size(0)
            if tmp > to:
                to = tmp
    if device is None:
        device = tensors[0].device
    shape = (len(tensors), to) + tensors[0].shape[1:]
    result = torch.full(shape, constant, dtype=tensors[0].dtype, device=device)
    for i in range(len(tensors)):
        tensor = tensors[i]
        result[i, :tensor.size(0)] = tensor
    return result


# @torch.jit.script
def pad_and_stack_2d(tensors, to_1= -1,
                     to_2= -1, constant= 0,
                     device=None):
    # type: (List[Tensor], int, int, int, Device) -> Tensor
    if to_1 == -1:
        for i in range(len(tensors)):
            tmp = tensors[i].size(0)
            if tmp > to_1:
                to_1 = tmp

    if to_2 == -1:
        for i in range(len(tensors)):
            tmp = tensors[i].size(1)
            if tmp > to_2:
                to_2 = tmp

    if device is None:
        device = tensors[0].device

    shape = (len(tensors), to_1, to_2) + tensors[0].shape[2:]
    result = torch.full(shape, constant, dtype=tensors[0].dtype, device=device)
    for i in range(len(tensors)):
        tensor = tensors[i]
        result[i, :tensor.size(0), :tensor.size(1)] = tensor
    return result


# @torch.jit.script
def pad_and_stack_2d_2(tensors_list, to_1= -1,
                       to_2= -1, constant= 0,
                       device=None):
    # type: (List[List[Tensor]], int, int, int, Device) -> Tensor
    if to_1 == -1 or to_2 == -1:
        new_to_1 = -1
        new_to_2 = -1
        for i in range(len(tensors_list)):
            inner_len = len(tensors_list[i])
            if inner_len > new_to_1:
                new_to_1 = inner_len
            for j in range(inner_len):
                tmp = tensors_list[i][j].size(0)
                if tmp > new_to_2:
                    new_to_2 = tmp
        if to_1 == -1:
            to_1 = new_to_1
        if to_2 == -1:
            to_2 = new_to_2

    if device is None:
        device = tensors_list[0][0].device

    shape = (len(tensors_list), to_1, to_2) + tensors_list[0][0].shape[1:]
    result = torch.full(shape, constant, dtype=tensors_list[0][0].dtype, device=device)
    for i in range(len(tensors_list)):
        for j in range(len(tensors_list[i])):
            tensor = tensors_list[i][j]
            result[i, j, :tensor.size(0)] = tensor
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
