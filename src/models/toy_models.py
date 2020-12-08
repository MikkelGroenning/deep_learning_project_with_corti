import random

import numpy as np
import torch
from src.data.toy import Continuous
from src.models.common import CriterionTrainer, VITrainer, VariationalInference
from src.models.iaf import VRAEIAF
from src.models.rae import RAE
from src.models.vrae import VRAE
from torch.nn import MSELoss
from torch.optim import Adam

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
data_parameters = {
    "max_length": 16,
    "min_length": 2,
    "error_scale": 0.05,
}


train_data = Continuous(num_observations=5_000, **data_parameters)
validation_data = Continuous(num_observations=2000, **data_parameters)
test_data = Continuous(num_observations=1000, **data_parameters)

batch_size = 100
max_epochs = 500

# Recurrent Autoencoder
rae = RAE(
    input_dim=1,
    latent_features=2,
    encoder_hidden_size=48,
    decoder_hidden_size=48,
)

# Variational Recurrent Autoencoder
vrae = VRAE(
    input_dim=1,
    latent_features=2,
    encoder_hidden_size=48,
    decoder_hidden_size=48,
)

# Variational Recurrent Autoencoder using IAF
vrae_iaf = VRAEIAF(
    input_dim=1,
    latent_features=2,
    encoder_hidden_size=48,
    decoder_hidden_size=48,
    flow_depth=4,
    flow_hidden_features=16, 
    flow_context_features=2,
)

if __name__ == "__main__":

    # Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.0005,
    }
    criterion = MSELoss(reduction="sum")
    optimizer = Adam(rae.parameters(), **optimizer_parameters)
    mt = CriterionTrainer(
        criterion=criterion,
        model=rae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=test_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "ToyRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.0005,
    }
    vi = VariationalInference()
    optimizer = Adam(vrae.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=vrae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=test_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "ToyVRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder using IAF
    optimizer_parameters = {
        "lr": 0.0005,
    }
    vi = VariationalInference()
    optimizer = Adam(vrae_iaf.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=vrae_iaf,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=test_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "ToyVRAEIAF"
    mt.restore_checkpoint()
    mt.train()
