from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

def get_loader(dataset, batch_size, pin_memory=False):

    sampler = BatchSampler(
        RandomSampler(dataset), batch_size=batch_size, drop_last=False
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        pin_memory=pin_memory,
    )