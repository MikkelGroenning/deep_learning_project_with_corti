from itertools import chain
from typing import Dict, Tuple

import numpy as np
import torch
from src.data.words import TwitterDataWords
from src.models.common import Decoder, EmbeddingPacked, Encoder, ReparameterizedDiagonalGaussian, VITrainer, simple_elementwise_apply
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import LSTM, Linear, Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam

from src.models.common import ModelTrainer
embedding_dimension = 300

# Encoder with 2 times output parameters
class _Encoder(Encoder):

    def __init__(
        self,
        input_dim,
        hidden_size_1,
        hidden_size_2,
        latent_features,
    ):
        super(_Encoder, self).__init__(
            input_dim=input_dim,
            hidden_size=hidden_size_1,
            latent_features=hidden_size_2,
            )

        self.linear = Linear(
            in_features=hidden_size_2, 
            out_features=2*latent_features
        )

    def forward(self, x):
        
        x = super(_Encoder, self).forward(x)
        x = self.linear(x)
        mu, log_sigma = x.chunk(2, dim=-1)
        return mu, log_sigma

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
        
        # Ensure flexibility with embedded input/output 
        if output_dim is None:
            self.output_dim=input_dim
        else:
            self.output_dim=output_dim

        self.encoder = _Encoder(
            input_dim=self.input_dim,
            hidden_size_1=self.encoder_hidden_size,
            hidden_size_2=self.encoder_hidden_size,
            latent_features=self.latent_features,
        )

        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size, 
            output_dim=2*self.output_dim,
        ) 

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * latent_features]))
        )

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""

        # compute the parameters of the posterior
        out = self.encoder(x)
        mu, log_sigma = out

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
    def __init__(self, input_dim, embedding_dim, *args, embedding=None, **kwargs):

        super().__init__(input_dim=embedding_dim, output_dim=input_dim, *args, **kwargs)

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

