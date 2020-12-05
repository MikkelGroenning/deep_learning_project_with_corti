
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import random
import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_sequence
from src.models.rae_words import RAEWords as RAE
from src.models.vrae_words import VRAEWords as VRAE
from src.models.iaf_words import IAFWords as IAF
from src.models.iaf_words import VariationalInference as VI_AIF

from src.models.vrae_chars import VariationalInference


from src.data.toy import ToyData, Continuous
from src.data.common import get_loader
from src.models.common import CriterionTrainer, OneHotPacked, VITrainer

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np

seed = 42
torch.set_deterministic(True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_parameters = {
    "max_length" : 16,
    "min_length" : 2,
    "error_scale" : 0.05,
}

batch_size = 100
max_epochs = 50

train_data = Continuous(num_observations=10_000, **data_parameters)
validation_data = Continuous(num_observations=1000, **data_parameters)
test_data = Continuous(num_observations=1000, **data_parameters)

num_tests = 1000
x_test = next(iter(get_loader(test_data, batch_size=num_tests)))
iaf = IAF(
    input_dim=1,
    latent_features=2,
    encoder_hidden_size=16,
    decoder_hidden_size=16,
    flow_depth=3,
    flow_hidden_features=8,
    flow_context_features=1,
)

optimizer_parameters = {
    "lr": 0.001,
}

class VITrainerNoCache(VITrainer):

    def save_checkpoint(self):
        pass

    def restore_checkpoint(self):
        pass

vi = VI_AIF(0.5)
optimizer = Adam(iaf.parameters(), **optimizer_parameters)

mt = VITrainerNoCache(
    vi=vi,
    model=iaf,
    optimizer=optimizer,
    batch_size=batch_size,
    max_epochs=max_epochs,
    training_data=train_data,
    validation_data=test_data,
)

mt.train(progress_bar='epoch')