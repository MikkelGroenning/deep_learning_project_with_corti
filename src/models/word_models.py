from os import X_OK
import random

import torch
from src.data.words import TwitterDataWords
from src.models.common import CriterionTrainer, VariationalInference, VITrainer
from src.models.iaf import VRAEIAF
from src.models.rae import RAE
from src.models.vrae import VRAE
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

import argparse

seed = 43

data = torch.load("data/processed/200316_embedding.pkl")
embedding_dim = 300

n_obs = len(data)
batch_size = 500
max_epochs = 500

indices = list(range(n_obs))
random.shuffle(indices)

num_train = int(0.6 * n_obs)
num_validation = int(0.2 * n_obs)
num_test = n_obs - num_train - num_validation

train_data = TwitterDataWords([data[i] for i in indices[:num_train]])
validation_data = TwitterDataWords([data[i] for i in indices[num_train:-num_test]])
test_data = TwitterDataWords([data[i] for i in indices[-num_test:]])

# Recurrent Autoencoder
word_rae = RAE(
    input_dim=embedding_dim,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
)

# Variational Recurrent Autoencoder
word_vrae = VRAE(
    input_dim=embedding_dim,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
)

# Variational Recurrent Autoencoder using IAF
word_vrae_iaf = VRAEIAF(
    input_dim=embedding_dim,
    latent_features=64,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
    flow_depth=6,
    flow_hidden_features=64,
    flow_context_features=8,
)


def train_rae(retrain=False):
    # Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.0005,
    }
    criterion = MSELoss(reduction="sum")
    optimizer = Adam(word_rae.parameters(), **optimizer_parameters)
    mt = CriterionTrainer(
        criterion=criterion,
        model=word_rae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "WordRAE"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


def train_vrae(retrain=False):

    # Variational Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.0005,
    }
    vi = VariationalInference()
    optimizer = Adam(word_vrae.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=word_vrae,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "WordVRAE"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


def train_vrae_iaf(retrain=False):

    # Variational Recurrent Autoencoder using IAF
    optimizer_parameters = {
        "lr": 0.0005,
    }
    vi = VariationalInference()
    optimizer = Adam(word_vrae_iaf.parameters(), **optimizer_parameters)
    mt = VITrainer(
        vi=vi,
        model=word_vrae_iaf,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=train_data,
        validation_data=validation_data,
        clip_max_norm=0.15,
    )
    mt.model_name = "WordVRAEIAF"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="?",
        help="models to train",
        default=["WordRAE", "WordVRAE", "WordVRAEIAF"],
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="retrain the models",
    )
    args = parser.parse_args()

    if "WordRAE" in args.models:
        train_rae(retrain=args.retrain)
    if "WordVRAE" in args.models:
        train_vrae(retrain=args.retrain)
    if "WordVRAEIAF" in args.models:
        train_vrae_iaf(retrain=args.retrain)
