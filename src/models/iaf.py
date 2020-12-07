
from math import log, pi

import torch
from src.models.common import (Decoder, EmbeddingPacked, Encoder,
                               ReparameterizedDiagonalGaussian,
                               )
from src.models.made import AutoRegressiveNN
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.nn import Module, Sequential


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

class VRAEIAF(Module):
    def __init__(
        self,
        input_dim,
        latent_features,
        encoder_hidden_size,
        decoder_hidden_size,
        flow_depth,
        flow_hidden_features,
        flow_context_features,
        output_dim=None,
    ):

        super(VRAEIAF, self).__init__()

        self.input_dim = input_dim
        self.latent_features = latent_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self.flow_depth = flow_depth
        self.flow_hidden_features = flow_hidden_features
        self.flow_context_features = flow_context_features

        self._init_encoder()
        self._init_decoder()

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

        # Define IAF for posterior
        self.iaf = IAF(
            *[
                IAFStep(
                    self.latent_features, 
                    self.flow_hidden_features, 
                    self.flow_context_features,
                )
                for _ in range(flow_depth)
            ]
        )

    def _init_encoder(self):

        self.encoder = Encoder(
            input_dim=self.input_dim,
            hidden_size=self.encoder_hidden_size,
            latent_features=2*self.latent_features + self.flow_context_features,
        )

    def _init_decoder(self):

        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size,
            output_dim=2 * self.output_dim,
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)

        h = h_x[:, -self.flow_context_features :]
        mu, log_sigma = h_x[:, : -self.flow_context_features].chunk(2, dim=-1)

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


class VRAEIAFWithEmbedder(VRAEIAF):


    def __init__(self, embedding_dim, input_dim, *args, embedding=None, **kwargs):

        super().__init__(*args, input_dim=embedding_dim, output_dim=input_dim, **kwargs)

        if embedding is None:
            self.embedding = EmbeddingPacked(
                num_embeddings=input_dim,
                embedding_dim=embedding_dim,
            )
        else:
            self.embedding = embedding
    
    def forward(self, x):

        x = self.embedding(x)
        x = super().forward(x)

        return x

    def _init_decoder(self):
        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size,
            output_dim=self.output_dim,
        )

    def observation_model(self, z: Tensor, batch_sizes: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        px_logits = self.decoder(z, batch_sizes) 

        return Categorical(logits=px_logits.data)
