import random
import torch
import numpy as np

from src.models.rae_words import RAEWords as RAE
from src.models.vrae_words import VRAEWords as VRAE
from src.models.iaf_words import IAFWords as IAF

from torch.nn import MSELoss
from src.models.vrae_words import VariationalInference as VI
from src.models.iaf_words import VariationalInference as VI_IAF


from src.data.common import get_loader
from src.models.common import CriterionTrainer, OneHotPacked, VITrainer
from src.data.toy import Continuous

from torch.optim import Adam

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
data_parameters = {
    "max_length" : 16,
    "min_length" : 2,
    "error_scale" : 0.05,
}


train_data = Continuous(num_observations=10_000, **data_parameters)
validation_data = Continuous(num_observations=2000, **data_parameters)
test_data = Continuous(num_observations=2000, **data_parameters)

batch_size = 100
max_epochs = 500

if __name__ == "__main__":

    # Recurrent Autoencoder
    rae = RAE(
        input_dim=1,
        latent_features=2,
        encoder_hidden_size=48,
        decoder_hidden_size=48,
    )
    optimizer_parameters = {
        "lr": 0.001,
    }
    criterion = MSELoss(reduction="sum")
    optimizer = Adam(rae.parameters(), **optimizer_parameters)
    mt = CriterionTrainer(
        criterion=criterion,
        model=rae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=200,
        training_data=train_data,
        validation_data=test_data,
    )
    mt.model_name = "ToyRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder
    vrae = VRAE(
        input_dim=1,
        latent_features=2,
        encoder_hidden_size=48,
        decoder_hidden_size=48,
    )
    optimizer_parameters = {
        "lr": 0.001,
    }
    vi = VI()
    optimizer = Adam(vrae.parameters(), **optimizer_parameters)
    mt = VITrainer( 
        vi=vi,
        model=vrae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=test_data,
    )
    mt.model_name = "ToyVRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder using IAF
    iaf = IAF(
        input_dim=1,
        latent_features=2,
        encoder_hidden_size=48,
        decoder_hidden_size=48,
        flow_depth=4,
        flow_hidden_features=24,
        flow_context_features=4,
    )
    optimizer_parameters = {
        "lr": 0.001,
    }
    vi = VI_IAF()
    optimizer = Adam(iaf.parameters(), **optimizer_parameters)
    mt = VITrainer( 
        vi=vi,
        model=iaf,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=test_data,
    )
    mt.model_name = "ToyIAF"
    mt.restore_checkpoint()
    mt.train()









