from itertools import chain
import numpy as np
import torch
from src.data.words import TwitterDataWords

from src.models.common import (
    CriteronTrainer, ModelTrainer,
    simple_elementwise_apply,
)

from torch.nn import LSTM, Module, MSELoss, Linear
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam


embedding_dimension = 300

class Encoder(Module):
    def __init__(
        self,
        latent_features=64,
        hidden_size=64,
        num_layers=2,
    ):

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_features = latent_features

        self.rnn = LSTM(
            input_size=embedding_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.ff = Linear(
            in_features=self.hidden_size, out_features=self.latent_features
        )

    def forward(self, x):

        x, (hidden_n, _) = self.rnn(x)

        return self.ff(hidden_n[-1])


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

        self.rnn = LSTM(
            input_size=self.latent_features,
            hidden_size=self.latent_features,
            num_layers=num_layers,
        )

        self.output_layer = Linear(hidden_size, embedding_dimension)

    def forward(self, x, batch_sizes):

        x = x.repeat(len(batch_sizes), 1, 1)

        lengths = -np.diff(np.append(batch_sizes.numpy(), 0))
        sequence_lengths = list(
            chain.from_iterable(n * [i + 1] for i, n in enumerate(lengths) if n)
        )[::-1]

        x = pack_padded_sequence(x, sequence_lengths)

        x, (_, _) = self.rnn(x)

        return simple_elementwise_apply(self.output_layer, x)


class RAEWords(Module):
    def __init__(self, latent_features=64):

        super(RAEWords, self).__init__()

        self.latent_features = latent_features

        self.encoder = Encoder(latent_features=self.latent_features)
        self.decoder = Decoder(latent_features=self.latent_features)

    def forward(self, x):

        batch_sizes = x.batch_sizes

        x = self.encoder(x)

        x = self.decoder(x, batch_sizes=batch_sizes)

        return x


# Default, should probably be explicit
model_parameters = {}

# Training parameters

batch_size = 2000
max_epochs = 500

optimizer_parameters = {"lr": 0.001}

if __name__ == "__main__":

    print("Loading dataset...")
    data = torch.load('data/processed/200316_embedding.pkl')

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDataWords(data[:split_idx])
    dataset_validation = TwitterDataWords(data[split_idx:])

    # dataset_train = TwitterDataWords(data[:1000])
    # dataset_validation = TwitterDataWords(data[1000:1500])

    criterion = MSELoss(reduction='sum')
    model = RAEWords(**model_parameters)
    optimizer = Adam(model.parameters(), **optimizer_parameters)

    mt = CriteronTrainer(
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
