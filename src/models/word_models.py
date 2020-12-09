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

data = torch.load("data/processed/200316_embedding.pkl")
embedding_dim = 300

batch_size = 128
max_epochs = 500

train_data = TwitterDataWords(data["train"])
validation_data = TwitterDataWords(data["validation"])
test_data = TwitterDataWords(data["test"])

# Recurrent Autoencoder
word_rae = RAE(
    input_dim=embedding_dim,
    latent_features=32,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
)

# Variational Recurrent Autoencoder
word_vrae = VRAE(
    input_dim=embedding_dim,
    latent_features=32,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
)

# Variational Recurrent Autoencoder using IAF
word_vrae_iaf = VRAEIAF(
    input_dim=embedding_dim,
    latent_features=32,
    encoder_hidden_size=64,
    decoder_hidden_size=64,
    flow_depth=6,
    flow_hidden_features=64,
    flow_context_features=8,
)


def train_rae(retrain=False):
    # Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.001,
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
        clip_max_norm=0.25,
    )
    mt.model_name = "WordRAE"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


def train_vrae(retrain=False):

    # Variational Recurrent Autoencoder
    optimizer_parameters = {
        "lr": 0.001,
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
        clip_max_norm=0.25,
    )
    mt.model_name = "WordVRAE"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


def train_vrae_iaf(retrain=False):

    # Variational Recurrent Autoencoder using IAF
    optimizer_parameters = {
        "lr": 0.001,
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
        clip_max_norm=0.25,
    )
    mt.model_name = "WordVRAEIAF"
    if not retrain:
        mt.restore_checkpoint()
    mt.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        help="model to train",
        default=None,
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="retrain the models",
    )
    args = parser.parse_args()

    if args.model is None:
        train_rae(retrain=args.retrain)
        train_vrae(retrain=args.retrain)
        train_vrae_iaf(retrain=args.retrain)
    elif "WordRAE" == args.model:
        train_rae(retrain=args.retrain)
    elif "WordVRAE" == args.model:
        train_vrae(retrain=args.retrain)
    elif "WordVRAEIAF" == args.model:
        train_vrae_iaf(retrain=args.retrain)

