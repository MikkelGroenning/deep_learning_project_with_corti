import torch
from src.models.common import (Decoder, EmbeddingPacked, Encoder,
                               ReparameterizedDiagonalGaussian)
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.nn import Module


class VRAE(Module):
    def __init__(
        self,
        input_dim,
        latent_features,
        encoder_hidden_size,
        decoder_hidden_size,
        output_dim=None,
    ):

        super(VRAE, self).__init__()

        self.input_dim = input_dim
        self.latent_features = latent_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self._init_encoder()
        self._init_decoder()

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def _init_encoder(self):

        self.encoder = Encoder(
            input_dim=self.input_dim,
            hidden_size=self.encoder_hidden_size,
            latent_features=2 * self.latent_features,
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


class VRAEWithEmbedder(VRAE):
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

        px_logits = self.decoder(z, batch_sizes)  # packedsequence

        return Categorical(logits=px_logits.data)
