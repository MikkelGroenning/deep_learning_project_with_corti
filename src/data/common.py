from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from pathlib import Path
import pandas as pd
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


num_train = 30_000
num_validation = 10_000
num_test = 9_000

path_hydrated = Path(f"data/interim/hydrated/{200316}.pkl")

data_full = pd.read_pickle(path_hydrated)
data_sampled = data_full.sample(num_train+num_validation+num_test, random_state=42)

data_train = data_sampled.iloc[:num_train, :]
data_validation = data_sampled.iloc[num_train:-num_test, :]
data_test = data_sampled.iloc[-num_test:, :]

data_trump = pd.read_pickle(Path(f"data/interim/hydrated/trump.pkl"))