import random

import pandas as pd
from src.data.characters import TwitterDataChars, alphabet
from src.models.common import CriterionTrainer, VariationalInference, VITrainer
from src.models.iaf import VRAEIAFWithEmbedder
from src.models.rae import RAEWithEmbedder
from src.models.vrae import VRAEWithEmbedder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

seed = 43

data = pd.read_pickle("data/interim/hydrated/200316.pkl")

n_obs = len(data)
batch_size = 256
max_epochs = 500

indices = list(range(n_obs))
random.shuffle(indices)

num_train = int(0.6 * n_obs)
num_validation = int(0.2 * n_obs)
num_test = n_obs - num_train - num_validation

train_data = TwitterDataChars(data.iloc[indices[:num_train]].copy())
validation_data = TwitterDataChars(data.iloc[indices[num_train:-num_test]].copy())
test_data = TwitterDataChars(data.iloc[indices[-num_test:]].copy())

# Recurrent Autoencoder
character_rae = RAEWithEmbedder(
    input_dim=len(alphabet),
    embedding_dim=8,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
)

# Variational Recurrent Autoencoder
character_vrae = VRAEWithEmbedder(
    input_dim=len(alphabet),
    embedding_dim=8,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
)

# Variational Recurrent Autoencoder using IAF
character_vrae_iaf = VRAEIAFWithEmbedder(
    input_dim=len(alphabet),
    embedding_dim=8,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
    flow_depth=6,
    flow_hidden_features=48,
    flow_context_features=8,
)

if __name__ == "__main__":

    # Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.001,
    }
    criterion = CrossEntropyLoss(reduction="sum")
    optimizer = Adam(character_rae.parameters(), **optimizer_parameters)
    mt = CriterionTrainer(
        criterion=criterion,
        model=character_rae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
    )
    mt.model_name = "CharacterRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.001,
    }
    vi = VariationalInference()
    optimizer = Adam(character_vrae.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=character_vrae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
    )
    mt.model_name = "CharacterVRAE"
    mt.restore_checkpoint()
    mt.train()

    # Variational Recurrent Autoencoder using IAF
    optimizer_parameters = {
        "lr": 0.001,
    }
    vi = VariationalInference()
    optimizer = Adam(character_vrae_iaf.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=character_vrae_iaf,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
    )
    mt.model_name = "CharacterVRAEIAF"
    mt.restore_checkpoint()
    mt.train()
