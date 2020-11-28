from abc import abstractmethod, ABC
from pathlib import Path
import sys
import numpy as np
from torch.nn.utils.rnn import PackedSequence
from torch.nn import Module, Embedding
import torch.nn.functional as FF
import torch

from tqdm import tqdm
from src.data.common import get_loader

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

    def forward(_, x):
        return simple_elementwise_apply(FF.one_hot, x)


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
    ):

        # Input parameters
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.training_data = training_data
        self.validation_data = validation_data

        # Data loaders
        self.train_loader = get_loader(self.training_data, batch_size)
        self.validation_loader = (
            get_loader(self.training_data, batch_size)
            if self.validation_data is not None
            else None
        )

        # Initializing fresh training params
        self.current_epoch = -1
        self.training_loss = []
        self.validation_loss = []

        # Saving model name
        self.model_name = model.__class__.__name__

        self.cuda = cuda
        if self.cuda:
            self.device = torch.device("cuda")
            print("Training using CUDA")
        else:
            self.device = torch.device("cpu")
            print("Training using CPU")

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
        for epoch in range(self.current_epoch + 1, self.max_epochs):

            # Track loss per batch
            epoch_training_loss = []
            epoch_validation_loss = []

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

            model.train()

            if progress_bar:
                train_loader = tqdm(self.train_loader)

            # For each sentence in training set
            for x in train_loader:

                x = get_variable(x)

                # Average loss per tweet
                loss = self.get_loss(x)

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
            self.training_loss.append(np.average(batch_average, weights=weigths))

            weigths, batch_average = zip(*epoch_validation_loss)
            self.validation_loss.append(np.average(batch_average, weights=weigths))

            self.current_epoch = epoch

            print(f"Epoch {epoch+1} done!")
            print(f"T. loss: {self.training_loss[-1]}")
            print(f"V. loss: {self.validation_loss[-1]}")
            sys.stdout.flush()

            self.save_checkpoint()

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


def get_trained_model(model_class, training_info=False):

    checkpoint = _get_checkpoint(model_class.__name__, device)
    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"])

    if training_info:
        return (
            model,
            {
                "training_loss": checkpoint["training_loss"],
                "validation_loss": checkpoint["validation_loss"],
                "num_epocs": checkpoint["epoch"]+1,
            },
        )
    else:
        return model
