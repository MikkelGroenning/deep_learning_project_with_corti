from itertools import chain
from typing import Dict, Tuple

import numpy as np
import torch
from src.data.words import TwitterDataWords
from src.models.common import (
    Decoder,
    Encoder,
    ModelTrainer,
    ParamEncoder,
    ReparameterizedDiagonalGaussian,
    VITrainer,
    simple_elementwise_apply,
)
from torch import Tensor
from src.models.made import AutoRegressiveNN

from torch.distributions import Distribution
from torch.nn import LSTM, Linear, Module, Sequential, Dropout, Linear, ReLU
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from math import pi, log

embedding_dimension = 300


class IAFStep(Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        context_features,
    ):

        super(IAFStep, self).__init__()

        self.m_nn = AutoRegressiveNN(
            in_features=in_features,
            hidden_features=hidden_features,
            context_features=context_features,
        )

        self.s_nn = AutoRegressiveNN(
            in_features=in_features,
            hidden_features=hidden_features,
            context_features=context_features,
        )

    def forward(self, z, h):

        m = self.m_nn(z, h)
        s = self.s_nn(z, h) + 1.5
        sigma = s.sigmoid()

        z = sigma * z + (1 - sigma) * m
        l = -torch.log(sigma + 1e-6).sum()

        return z, l


class IAF(Sequential):
    def forward(self, mu, log_sigma, h):

        eps = torch.empty_like(mu).normal_()
        z = log_sigma.exp() * eps + mu
        l = -torch.sum(log_sigma + 1 / 2 * torch.pow(eps, 2) + 1 / 2 * log(2 * pi))

        for module in self:
            z, log_lik = module(z, h)
            l += log_lik

        return z, l


class Encoder_(Encoder):
    def __init__(
        self,
        input_dim,
        hidden_size_1,
        hidden_size_2,
        latent_features,
        context_features,
    ):
        super(Encoder_, self).__init__(
            input_dim=input_dim,
            hidden_size=hidden_size_1,
            latent_features=hidden_size_2,
        )

        self.context_features = context_features
        self.linear = Linear(
            in_features=hidden_size_2,
            out_features=2 * latent_features + context_features,
        )

    def forward(self, x):

        x = super(Encoder_, self).forward(x)
        x = self.linear(x)

        h = x[:, -self.context_features :]
        mu, log_sigma = x[:, : -self.context_features].chunk(2, dim=-1)

        return mu, log_sigma, h


class IAFWords(Module):
    def __init__(
        self,
        input_dim,
        latent_features,
        encoder_hidden_size,
        decoder_hidden_size,
        flow_depth,
        flow_hidden_features,
        flow_context_features,
    ):

        super(IAFWords, self).__init__()

        self.input_dim = input_dim
        self.latent_features = latent_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = Encoder_(
            input_dim=self.input_dim,
            hidden_size_1=self.encoder_hidden_size,
            hidden_size_2=self.encoder_hidden_size,
            latent_features=self.latent_features,
            context_features=flow_context_features,
        )

        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size,
            output_dim=2 * self.input_dim,
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

        # Define IAF for posterior
        self.iaf = IAF(
            *[
                IAFStep(
                    self.latent_features, flow_hidden_features, flow_context_features
                )
                for _ in range(flow_depth)
            ]
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        mu, log_sigma, h = self.encoder(x)

        z, lz = self.iaf(mu, log_sigma, h)

        return z, lz

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
        z, lz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.batch_sizes[0])

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z, batch_sizes=x.batch_sizes)

        return {"px": px, "pz": pz,  "z": z, "lz": lz}


class VariationalInference(Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: Module, x: Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        px, pz, z, lz = [outputs[k] for k in ["px", "pz", "z", "lz"]]

        log_px = px.log_prob(x.data).sum() / len(z)
        log_pz = pz.log_prob(z).mean()

        log_qz = lz / len(z)

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
    data = torch.load("data/processed/200316_embedding.pkl")

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDataWords(data[:split_idx])
    dataset_validation = TwitterDataWords(data[split_idx:])

    # dataset_train = TwitterDataWords(data[:1000])
    # dataset_validation = TwitterDataWords(data[1000:1500])

    vi = VariationalInference()
    model = IAFWords(**model_parameters)
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
