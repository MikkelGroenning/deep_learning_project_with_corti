from itertools import chain
from typing import Dict, Tuple

import numpy as np
import torch
from src.data.words import TwitterDataWords
from src.models.common import VITrainer, simple_elementwise_apply
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import LSTM, Linear, Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam

from src.models.common import ModelTrainer
embedding_dimension = 300

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

        # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
        # Note the 2*latent_features
        self.ff = Linear(
            in_features=self.hidden_size, out_features=2 * self.latent_features
        )

    def forward(self, x):

        x, (hidden_n, _) = self.rnn(x)

        h_x = self.ff(hidden_n[-1])
        mu, log_sigma = h_x.chunk(2, dim=-1)

        return mu, log_sigma


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

        self.output_layer = Linear(hidden_size, 2*embedding_dimension)

    def forward(self, x, batch_sizes):

        # waste some memory, but not much
        x = x.repeat(len(batch_sizes), 1, 1)

        # And now for the tricky part
        # (calculating sequence lengths from batch sizes)
        lengths = -np.diff(np.append(batch_sizes.numpy(), 0))
        sequence_lengths = list(
            chain.from_iterable(n * [i + 1] for i, n in enumerate(lengths) if n)
        )[::-1]

        x = pack_padded_sequence(x, sequence_lengths)

        x, (_, _) = self.rnn(x)

        return simple_elementwise_apply(self.output_layer, x)


class VRAEWords(Module):

    def __init__(self, latent_features=64):

        super(VRAEWords, self).__init__()

        self.latent_features = latent_features

        self.encoder = Encoder(self.latent_features)
        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=64,
            num_layers=2,
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        mu, log_sigma = self.encoder(x)

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

        h_z = self.decoder(z, batch_sizes) 
        mu, log_sigma = h_z.data.chunk(2, dim=-1)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)

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

        log_px = px.log_prob(x.data).sum() / len(z)

        log_pz = pz.log_prob(z).mean()
        log_qz = qz.log_prob(z).mean()

        # compute the ELBO with and without the beta parameter:
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz

        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl

        loss = -beta_elbo

        # prepare the output
        with torch.no_grad():
            diagnostics = {"elbo": elbo, "log_px": log_px, "kl": kl}

        return loss, diagnostics, outputs

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

    vi = VariationalInference()
    model = VRAEWords(**model_parameters)
    optimizer = Adam(model.parameters(), **optimizer_parameters)

    mt = VITrainer(
        vi=vi,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=dataset_train,
        validation_data=dataset_validation,
    )

    mt.restore_checkpoint()
    mt.train()
