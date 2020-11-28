from torch.utils.data import dataloader
from torch.utils.data.sampler import BatchSampler, RandomSampler

def get_loader(dataset, batch_size, pin_memory=False):

    sampler = BatchSampler(
        RandomSampler(dataset), batch_size=batch_size, drop_last=False
    )
    return dataloader(
        dataset,
        batch_size=None,
        sampler=sampler,
        pin_memory=pin_memory,
    )