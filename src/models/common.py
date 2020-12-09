import sys
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as FF
from src.data.common import get_loader
from torch.distributions import Distribution
from torch.nn import LSTM, Embedding, Linear, Module
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch.tensor import Tensor
from tqdm import tqdm, trange

from torch.nn.utils import clip_grad_norm_

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_directory = Path(__file__).parent / ".." / ".." / "models"


def simple_elementwise_apply(fn, packed_sequence):
    """applies a pointwise function fn to each element in packed_sequence"""
    return PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


class OneHotPacked(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return PackedSequence(
            FF.one_hot(x.data, num_classes=self.num_classes).float(), 
            x.batch_sizes
            )

class EmbeddingPacked(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding = Embedding(**kwargs)

    def forward(self, x):
        return simple_elementwise_apply(self.embedding, x)


class ModelTrainer(ABC):
    def __init__(
        self,
        model,
        optimizer,
        max_epochs,
        batch_size,
        training_data,
        validation_data=None,
        clip_max_norm=None,
    ):

        # Input parameters
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.training_data = training_data
        self.validation_data = validation_data

        self.clip_max_norm=clip_max_norm

        # Data loaders
        self.train_loader = get_loader(self.training_data, batch_size)
        self.validation_loader = (
            get_loader(self.validation_data, batch_size)
            if self.validation_data is not None
            else None
        )

        # Initializing fresh training params
        self.current_epoch = -1
        self.training_loss = []
        self.validation_loss = []

        self.best_model = {
            'validation_loss' : float('inf'),
            'state_dict' : None,
        }


        # Saving model name
        self.model_name = model.__class__.__name__

        self.cuda = cuda
        if self.cuda:
            self.device = torch.device("cuda")
            print("Training using CUDA")
        else:
            self.device = torch.device("cpu")
            print("Training using CPU")

        sys.stdout.flush()

        self.model.to(device)

    @abstractmethod
    def get_loss(self, x):
        """ Get average loss in batch x. x is PackedSequence """
        pass

    def train(self, progress_bar=False):

        model = self.model
        train_loader = self.train_loader
        validation_loader = self.validation_loader

        optimizer = self.optimizer

        # For each epoch
        if progress_bar == 'epoch':
            epoch_iter = trange(self.current_epoch + 1, self.max_epochs)
        else:
            epoch_iter = range(self.current_epoch + 1, self.max_epochs)

        for epoch in epoch_iter:

            # Track loss per batch
            epoch_training_loss = []
            epoch_validation_loss = []

            model.train()

            if progress_bar == 'batch':
                train_loader = tqdm(self.train_loader)

            # For each sentence in training set
            for x in train_loader:

                x = get_variable(x)

                # Average loss per tweet
                loss = self.get_loss(x)

                optimizer.zero_grad()
                loss.backward()

                if self.clip_max_norm is not None:
                    clip_grad_norm_(model.parameters(), self.clip_max_norm)

                optimizer.step()

                epoch_training_loss.append(
                    (
                        x.batch_sizes[0].numpy(),
                        get_numpy(loss.detach()),
                    )
                )

            model.eval()

            with torch.no_grad():

                # For each sentence in validation set
                for x in validation_loader:

                    x = get_variable(x)
                    loss = self.get_loss(x)

                    # Update loss
                    epoch_validation_loss.append(
                        (
                            x.batch_sizes[0].numpy(),
                            get_numpy(loss.detach()),
                        )
                    )

            # Save loss for plot
            weigths, batch_average = zip(*epoch_training_loss)
            self.training_loss.append(np.average(batch_average, weights=weigths))

            weigths, batch_average = zip(*epoch_validation_loss)
            self.validation_loss.append(np.average(batch_average, weights=weigths))

            if self.validation_loss[-1] < self.best_model['validation_loss']:
                self.best_model['validation_loss'] = self.validation_loss[-1]
                self.best_model['state_dict'] = self.model.state_dict()

            self.current_epoch = epoch

            if progress_bar != 'epoch':
                print(f"Epoch {epoch+1} done!")
                print(f"T. loss: {self.training_loss[-1]}")
                print(f"V. loss: {self.validation_loss[-1]}")
                sys.stdout.flush()
            elif progress_bar == 'epoch':
                epoch_iter.set_postfix({
                    "t_loss" : self.training_loss[-1],
                    "v_loss" : self.validation_loss[-1],
                })

            self.save_checkpoint()

        (model_directory / self.model_name / "finished").touch()

    def save_checkpoint(self):

        loc = model_directory / self.model_name
        loc.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_loss": self.training_loss,
                "validation_loss": self.validation_loss,
                "best_model": self.best_model,
            },
            loc / "checkpoint.pt",
        )

    def restore_checkpoint(self):

        checkpoint = _get_checkpoint(self.model_name, self.device)

        if checkpoint is None:
            print("No checkpoint found, training fresh model.")
            return

        print("Checkpoint found, continuing training.")

        self.current_epoch = checkpoint["epoch"]
        self.training_loss = checkpoint["training_loss"]
        self.validation_loss = checkpoint["validation_loss"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_model = checkpoint["best_model"]

        if self.cuda:
            self.model.to(self.device)
            # Fix for optimizer on gpu
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


def _get_checkpoint(model_name, device):

    try:
        checkpoint = torch.load(
            model_directory / model_name / "checkpoint.pt",
            map_location=device,
        )
        return checkpoint
    except FileNotFoundError:
        return None


def get_trained_model(model, training_info=False, model_name=None):

    if model_name is None:
        model_name = model.__class__.__name__

    checkpoint = _get_checkpoint(model_name, device)
    model.load_state_dict(checkpoint["best_model"]["state_dict"])

    if training_info:
        return (
            model,
            {
                "training_loss": checkpoint["training_loss"],
                "validation_loss": checkpoint["validation_loss"],
                "best_validation_loss": checkpoint["best_model"]["validation_loss"],
                "num_epocs": checkpoint["epoch"] + 1,
            },
        )
    else:
        return model


def decode_tweet_to_text(decoded_tweet, embedding, joined=False):
    # Init list for words
    decoded_tweet_word_list = []

    # Loop over all words
    for word in decoded_tweet:
        # Stop when end reached
        if all(word == 0):
            break

        # Add decoded word
        decoded_tweet_word_list.append(
            embedding.similar_by_vector(np.array(word), topn=1, restrict_vocab=None)[0][
                0
            ]
        )

    # Return decoded list
    if joined:
        return " ".join(decoded_tweet_word_list)
    else:
        return decoded_tweet_word_list


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def decode_tweet_to_text(decoded_tweet, embedding, joined=False):
    # Init list for words
    decoded_tweet_word_list = []

    # Loop over all words
    for word in decoded_tweet:
        # Stop when end reached
        if all(word == 0):
            break

        # Add decoded word
        decoded_tweet_word_list.append(
            embedding.similar_by_vector(np.array(word), topn=1, restrict_vocab=None)[0][
                0
            ]
        )

    # Return decoded list
    if joined:
        return " ".join(decoded_tweet_word_list)
    else:
        return decoded_tweet_word_list


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def actual_decode_similarity(sample, target, embedding):
    # Calculate average
    rsum = np.zeros(300)

    for word, _ in embedding.vocab.items():
        rsum += embedding[word]

    average = rsum / len(embedding.vocab)

    # Init values
    cos_sim_embed = 0
    cos_sim_target = 0
    cos_sim_avg = 0
    counter = 0

    # Loop over words
    for (sample_word, target_word) in zip(sample, target):
        # Stop when end reached
        if all(sample_word == 0):
            break

        # Get debedded word
        embed_word = embedding.similar_by_vector(
            np.array(sample_word), topn=1, restrict_vocab=None
        )[0][0]

        # Calculate cosine similarities
        cos_sim_embed += cos_sim(embedding[embed_word], sample_word)
        cos_sim_target += cos_sim(target_word, sample_word)
        cos_sim_avg += cos_sim(target_word, average)

        # Increment couter for average
        counter += 1

    return {
        embedding: cos_sim_embed / counter,
        target: cos_sim_target / counter,
        average: cos_sim_avg / counter,
    }


class VITrainer(ModelTrainer):
    def __init__(self, vi, *args, **kwargs):

        super(VITrainer, self).__init__(*args, **kwargs)
        self.vi = vi

    def get_loss(self, x):

        loss, _, _ = self.vi(self.model, x)

        return loss


class CriterionTrainer(ModelTrainer):
    def __init__(self, criterion, *args, **kwargs):

        super(CriterionTrainer, self).__init__(*args, **kwargs)
        self.criterion = criterion

    def get_loss(self, x):

        output = self.model(x)
        loss = self.criterion(output.data, x.data) / x.batch_sizes[0]

        return loss

# Encoder defition
class Encoder(Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        latent_features,
    ):

        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_features = latent_features

        self.rnn = LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
        )

        self.linear = Linear(
            in_features=self.hidden_size,
            out_features=self.latent_features,
            bias=False,
        )

    def forward(self, x):

        x, (hidden_n, _) = self.rnn(x)
        x = self.linear(hidden_n[-1])

        return x



# Decoder defitinion
class Decoder(Module):
    def __init__(
        self,
        latent_features,
        hidden_size,
        output_dim,
    ):
        super(Decoder, self).__init__()

        self.latent_features = latent_features
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.rnn1 = LSTM(
            input_size=self.latent_features,
            hidden_size=self.latent_features,
        )

        self.rnn2 = LSTM(
            input_size=self.latent_features,
            hidden_size=self.hidden_size,
        )

        self.output_layer = Linear(hidden_size, self.output_dim )

    def forward(self, x, batch_sizes):

        x = x.repeat(len(batch_sizes), 1, 1)

        lengths = -np.diff(np.append(batch_sizes.numpy(), 0))
        sequence_lengths = list(
            chain.from_iterable(n * [i + 1] for i, n in enumerate(lengths) if n)
        )[::-1]

        x = pack_padded_sequence(x, sequence_lengths)

        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)

        return simple_elementwise_apply(self.output_layer, x)


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


class VariationalInference(Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: Module, x: Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs

        pz = outputs['pz'] # Prior
        z = outputs['z'] # Sample from posterior
        px = outputs['px'] # Observation model

        if 'lz' in outputs:
            lz = outputs['lz'] # Log likelihood of sample from approx. posterior
            log_qz = lz / len(z)
        else:
            qz = outputs['qz'] # Approx. posterior
            log_qz = qz.log_prob(z).sum(dim=1).mean()


        log_px = px.log_prob(x.data).sum() / len(z)
        log_pz = pz.log_prob(z).sum(dim=1).mean()

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

