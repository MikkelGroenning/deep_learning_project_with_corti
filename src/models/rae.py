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
    EmbeddingPacked,
    get_checkpoint,
    get_numpy,
    get_variable, save_checkpoint,
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


model_name = "RecurrentAutoencoder"
# Default, should probably be explicit
model_parameters = {}

# Training parameters

batch_size = 2000
max_epochs = 10

optimizer_parameters = {"lr": 0.001}

def get_model():

    try:
        checkpoint = get_checkpoint(model_name)
    except FileNotFoundError:
        print("Model not trained yet")
        return None

    model = RecurrentAutoencoder(**model_parameters)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDatasetChar(data.iloc[:split_idx, :].copy())
    dataset_validation = TwitterDatasetChar(data.iloc[split_idx:, :].copy())

    if cuda:
        print("Using CUDA...")
    else:
        print("Using CPU...")
    sys.stdout.flush()

    criterion = CrossEntropyLoss(reduction="sum")

    model = RecurrentAutoencoder(**model_parameters)
    optimizer = Adam(model.parameters(), **optimizer_parameters)

    checkpoint = get_checkpoint(model_name)

    if checkpoint is not None:
        print("Continuing from previously trained model")
        current_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        training_loss = checkpoint["training_loss"]
        validation_loss = checkpoint["validation_loss"]
    else:
        current_epoch = -1
        training_loss = []
        validation_loss = []

    train_loader = get_loader(dataset_train, batch_size, pin_memory=cuda)
    validation_loader = get_loader(dataset_validation, batch_size, pin_memory=cuda)

    if cuda:
        model.to(torch.device("cuda"))
        # Fix for optimizer on cpu
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # For each epoch
    for epoch in range(current_epoch + 1, max_epochs):

        # Track loss per batch
        epoch_training_loss = []
        epoch_validation_loss = []

        model.eval()

        with torch.no_grad():

            # For each sentence in validation set
            for x in validation_loader:

                x = get_variable(x)

                output = model(x)
                
                # Average loss per tweet 
                loss = criterion(output.data, x.data) // x.batch_sizes[0]

                # Update loss
                epoch_validation_loss.append(
                    (
                        x.batch_sizes[0].numpy(),
                        get_numpy(loss.detach()),
                    )
                )

        model.train()

        # For each sentence in training set
        for x in train_loader:

            x = get_variable(x)

            # Forward pass
            output = model(x)
            
            # Average loss per tweet 
            loss = criterion(output.data, x.data) / x.batch_sizes[0]


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_training_loss.append(
                (
                    x.batch_sizes[0].numpy(),
                    get_numpy(loss.detach()),
                )
            )

        # Save loss for plot
        weigths, batch_average = zip(*epoch_training_loss)
        training_loss.append(np.average(batch_average, weights=weigths))

        weigths, batch_average = zip(*epoch_validation_loss)
        validation_loss.append(np.average(batch_average, weights=weigths))

        print(f"Epoch {epoch+1} done!")
        print(f"T. loss: {training_loss[-1]}")
        print(f"V. loss: {validation_loss[-1]}")
        sys.stdout.flush()

        save_checkpoint(
            model_name=model_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            training_loss=training_loss,
            validation_loss=validation_loss,
        )