from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from itertools import chain
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
from src.data.data_loader import (
    TwitterDataset,
    alphabet,
    get_loader,
    character_to_number,
)
from src.models.common import (
    EmbeddingPacked,
    get_numpy,
    get_variable,
    simple_elementwise_apply,
    cuda,
)

from torch import Tensor
from torch.nn import LSTM, CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
from torch.optim import SGD, Adam
from torch.distributions import Distribution, Bernoulli

num_classes = len(alphabet)
latent_features = 64
embedding_dim = 10
max_length = 300


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert (
            mu.shape == log_sigma.shape
        ), f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return torch.distributions.normal.Normal(self.mu, self.sigma).log_prob(z)


class Encoder(Module):
    def __init__(
        self,
        embedding_dim=embedding_dim,
        latent_features=latent_features,
        hidden_dim=64,
        num_layers=2,
    ):

        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_features = latent_features

        self.embedding = EmbeddingPacked(
            num_embeddings=num_classes,
            embedding_dim=self.embedding_dim,
        )

        self.rnn = LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
        # Note the 2*latent_features
        self.ff = Linear(
            in_features=self.hidden_dim, out_features=2 * self.latent_features
        )

    def forward(self, x):

        x = self.embedding(x)
        x, (hidden_n, _) = self.rnn(x)

        return self.ff(hidden_n[-1])


# Model defition
class Decoder(Module):
    def __init__(
        self,
        latent_features=latent_features,
        hidden_size=64,
        max_length=max_length,
        num_layers=2,
    ):
        super(Decoder, self).__init__()

        self.max_length = max_length
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

        # waste some memory, but not much
        x = x.repeat(len(batch_sizes), 1, 1)

        # And now for the tricky part
        lengths = -np.diff(np.append(batch_sizes.numpy(), 0))
        sequence_lengths = list(
            chain.from_iterable(n * [i + 1] for i, n in enumerate(lengths) if n)
        )[::-1]

        x = pack_padded_sequence(x, sequence_lengths)

        x, (_, _) = self.rnn(x)

        return simple_elementwise_apply(self.output_layer, x)


class RecurrentVariationalAutoencoder(Module):
    def __init__(self, latent_features=64, max_length=300):

        super(RecurrentVariationalAutoencoder, self).__init__()

        self.latent_features = latent_features

        self.encoder = Encoder(self.latent_features)
        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=64,
            max_length=max_length,
            num_layers=2,
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int = 1) -> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(
            batch_size, *self.prior_params.shape[-1:]
        )
        mu, log_sigma = prior_params.chunk(2, dim=-1)

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: Tensor, batch_sizes: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        px_logits = self.decoder(z, batch_sizes)
        return Categorical(logits=px_logits)

    def forward(self, x):

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.batch_sizes[0])

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z, batch_sizes=x.batch_sizes)

        return {"px": px, "pz": pz, "qz": qz, "z": z}


class VariationalInference(Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: Module, x: Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # evaluate log probabilities
        x_padded, batch_sizes = pad_packed_sequence(
            x, total_length=300, padding_value=character_to_number["P"]
        )
        log_px = px.log_prob(x_padded).sum(dim=0)
        log_pz = pz.log_prob(z).sum(dim=1)
        log_qz = qz.log_prob(z).sum(dim=1)

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz

        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl

        loss = -beta_elbo.sum()

        # prepare the output
        with torch.no_grad():
            diagnostics = {"elbo": elbo, "log_px": log_px, "kl": kl}

        return loss, diagnostics, outputs


if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDataset(data.iloc[:split_idx, :].copy())
    dataset_validation = TwitterDataset(data.iloc[split_idx:, :].copy())

    if cuda:
        print("Using CUDA...")
    else:
        print("Using CPU...")

    batch_size = 5000
    train_loader = get_loader(dataset_train, batch_size, pin_memory=cuda)
    validation_loader = get_loader(dataset_validation, batch_size, pin_memory=cuda)

    net = RecurrentVariationalAutoencoder()

    if cuda:
        net = net.cuda()

    # Hyper-parameters
    num_epochs = 10

    # Define a loss function and optimizer for this problem
    criterion = CrossEntropyLoss(ignore_index=-1)
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Track loss
    training_loss, validation_loss = [], []

    # For each epoch
    for i in range(num_epochs):

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        net.eval()

        with torch.no_grad():

            # For each sentence in validation set
            for x in validation_loader:

                # One-hot encode input and target sequence
                x = get_variable(x)

                # Forward pass
                output = net(x)
                # Backward pass
                loss = criterion(
                    output.view(-1, num_classes),
                    pad_packed_sequence(x, total_length=300, padding_value=-1)[0].view(
                        -1
                    ),
                )
                # Update loss
                epoch_validation_loss += get_numpy(loss.detach())

        net.train()

        # For each sentence in training set
        for x in train_loader:

            x = get_variable(x)

            # Forward pass
            output = net(x)
            # Backward pass
            loss = criterion(
                output.view(-1, 57),
                pad_packed_sequence(x, total_length=300, padding_value=-1)[0].view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_training_loss += get_numpy(loss.detach())

        print(f"Epoch {i+1} done!")
        # Save loss for plot
        training_loss.append(epoch_training_loss / len(dataset_train))
        validation_loss.append(epoch_validation_loss / len(dataset_validation))

    model_name = net.__class__.__name__
    model_directory = Path(f"./models/{model_name}/")
    model_directory.mkdir(parents=True, exist_ok=True)

    time_string = datetime.now().strftime("%y%m%d_%H%M%S")

    with open(model_directory / f"{time_string}_results.json", "w+") as f:
        json.dump(
            {"training_loss": training_loss, "validation loss": validation_loss}, f
        )

    torch.save(net.state_dict(), model_directory / f"{time_string}_state_dict.pt")
