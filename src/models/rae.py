from itertools import chain
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import pandas as pd
import torch
from src.data.characters import TwitterDatasetChar, alphabet
from src.data.common import get_loader

from src.models.common import (
    EmbeddingPacked, ModelTrainer,
    simple_elementwise_apply,
    cuda,
)

from torch.nn import LSTM, CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
from torch.optim import SGD
from torch.optim import Adam

num_classes = len(alphabet)


class Encoder(Module):
    def __init__(
        self,
        embedding_dim=10,
        latent_features=64,
        hidden_size=64,
        num_layers=2,
    ):

        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_features = latent_features

        self.embedding = EmbeddingPacked(
            num_embeddings=num_classes,
            embedding_dim=self.embedding_dim,
        )

        self.rnn = LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.ff = Linear(
            in_features=self.hidden_size, out_features=self.latent_features
        )

    def forward(self, x):

        x = self.embedding(x)
        x, (hidden_n, _) = self.rnn(x)

        return hidden_n[-1]


# Model defition
class Decoder(Module):
    def __init__(
        self,
        latent_features=64,
        hidden_size=64,
        num_layers=2,
    ):
        super(Decoder, self).__init__()

        self.latent_features = latent_features
        self.hidden_size = hidden_size
        self.n_features = len(alphabet)

        self.rnn = LSTM(
            input_size=self.latent_features,
            hidden_size=self.latent_features,
            num_layers=num_layers,
        )

        self.output_layer = Linear(hidden_size, self.n_features)

    def forward(self, x, batch_sizes):

        x = x.repeat(len(batch_sizes), 1, 1)

        lengths = -np.diff(np.append(batch_sizes.numpy(), 0))
        sequence_lengths = list(
            chain.from_iterable(n * [i + 1] for i, n in enumerate(lengths) if n)
        )[::-1]

        x = pack_padded_sequence(x, sequence_lengths)

        x, (_, _) = self.rnn(x)

        return simple_elementwise_apply(self.output_layer, x)


class RecurrentAutoencoder(Module):
    def __init__(self, latent_features=64):

        super(RecurrentAutoencoder, self).__init__()

        self.latent_features = latent_features

        self.encoder = Encoder(latent_features=self.latent_features)
        self.decoder = Decoder(latent_features=self.latent_features)

    def forward(self, x):

        batch_sizes = x.batch_sizes

        x = self.encoder(x)

        x = self.decoder(x, batch_sizes=batch_sizes)

        return x


class RAETrainer(ModelTrainer):

    def __init__(self, criterion, *args, **kwargs):

        super(RAETrainer, self).__init__(*args, **kwargs)
        self.criterion = criterion

    def get_loss(self, x):

        output = self.model(x)
        loss = self.criterion(output.data, x.data) / x.batch_sizes[0]

        return loss

# Default, should probably be explicit
model_parameters = {}

# Training parameters

batch_size = 2000
max_epochs = 500

optimizer_parameters = {"lr": 0.001}

if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDatasetChar(data.iloc[:split_idx, :].copy())
    dataset_validation = TwitterDatasetChar(data.iloc[split_idx:, :].copy())

    criterion = CrossEntropyLoss(reduction="sum")
    model = RecurrentAutoencoder(**model_parameters)
    optimizer = Adam(model.parameters(), **optimizer_parameters)

    mt = RAETrainer(
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=dataset_train,
        validation_data=dataset_validation,
    )

    mt.restore_checkpoint()
    mt.train()
