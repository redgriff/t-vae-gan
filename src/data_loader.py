import torch


class DataLoader:
    def __init__(self, *tensors, batch_size: int = 32, shuffle: bool = False):
        self.tensors = tensors
        self.data_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(self.data_len)
        if self.shuffle:
            indices = torch.randperm(self.data_len)

        for start_idx in range(0, self.data_len, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.data_len)
            yield tuple(t[indices[start_idx:end_idx]] for t in self.tensors)

    def __len__(self):
        return (self.data_len + self.batch_size - 1) // self.batch_size
