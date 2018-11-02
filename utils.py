import torch


class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """

    def __init__(self, batch_idxs_np, device=torch.device("cpu")):
        self.batch_idxs_np = batch_idxs_np
        # Note that the torch copy will be on GPU if use_cuda is set
        self.batch_idxs = torch.from_numpy(batch_idxs_np).to(device)
        self.batch_size = self.batch_idxs.max() + 1

        negative_one = torch.tensor([-1], device=device)
        batch_idxs_extra = torch.cat([negative_one, self.batch_idxs, negative_one])
        self.boundaries = torch.ne(batch_idxs_extra[1:], batch_idxs_extra[:-1]).nonzero().view(-1)
        self.startpoints = self.boundaries[:-1]
        self.endpoints = self.boundaries[1:] - 1
        self.seq_lens = self.boundaries[1:] - self.boundaries[:-1]
        assert self.seq_lens.size(0) == self.batch_size
        self.max_len = torch.max(self.boundaries[1:] - self.boundaries[:-1])


def convert_to_torch_tensor(inputs, device=None):
    for i in inputs.keys():
        inputs[i] = torch.tensor(inputs[i], device=device)


def smart_from_numpy(ndarray, device=torch.device("cpu")):
    return torch.from_numpy(ndarray).pin_memory().to(device)
