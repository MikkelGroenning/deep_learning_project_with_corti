
import random
import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from src.models.rae_words import RAEWords as RAE
from src.models.vrae_words import VRAEWords as VRAE
from src.models.iaf_words import IAFWords as IAF

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
    "max_length" : 30,
    "min_length" : 2,
    "error_scale" : 0.02,
}

batch_size = 100
max_epochs = 50

train_data = Continuous(num_observations=10_000, **data_parameters)
validation_data = Continuous(num_observations=1_000, **data_parameters)
test_data = Continuous(num_observations=1_000, **data_parameters)

vrae = VRAE(
    input_dim=1,
    latent_features=2,
    encoder_hidden_size=32,
    decoder_hidden_size=32,
)

optimizer_parameters = {
    "lr": 0.01,
}

vi = VariationalInference()
optimizer = Adam(vrae.parameters(), **optimizer_parameters)

class VITrainerNoCache(VITrainer):

    def save_checkpoint(self):
        pass

    def restore_checkpoint(self):
        pass

mt = VITrainerNoCache(
    vi=vi,
    model=vrae,
    optimizer=optimizer,
    batch_size=batch_size,
    max_epochs=max_epochs,
    training_data=train_data,
    validation_data=test_data,
)

mt.train(progress_bar=True)