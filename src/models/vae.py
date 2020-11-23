import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from src.data.data_loader import TwitterDataset, alphabet, get_loader
from src.models.common import (EmbeddingPacked, get_numpy, get_variable,
                               simple_elementwise_apply, cuda)

from torch.nn import LSTM, CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.optim import SGD

num_classes = len(alphabet)

class Encoder(Module):

    def __init__(self, hidden_dim=64, num_layers=2):

        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.embedding = EmbeddingPacked(
            num_embeddings = num_classes,
            embedding_dim = 10,
        )

        self.rnn = LSTM(
            input_size=10,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
        )

    def forward(self, x):

        x = self.embedding(x)
        x, (hidden_n, _) = self.rnn(x)

        return hidden_n[-1]

# Model defition
class Decoder(Module):
    
    def __init__(self, input_dim=64, max_length=300, num_layers=2):
        super(Decoder, self).__init__()

        self.max_length = max_length
        self.input_dim = input_dim
        self.n_features = len(alphabet)

        self.rnn = LSTM(
            input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_layers=num_layers
        )

        self.output_layer = Linear(self.input_dim, self.n_features)

    def forward(self, x):

        x = x.repeat(self.max_length, 1, 1)

        x, (_, _) = self.rnn(x)

        return self.output_layer(x)

class RecurrentVariationalAutoencoder(Module):

    def __init__(self, embedding_dim=64, max_length = 300):

        super(RecurrentVariationalAutoencoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.embedding_dim)
        self.decoder = Decoder(self.embedding_dim, max_length)

        self.hidden2mean = Linear(self.embedding_dim, self.embedding_dim)
        self.hidden2logv = Linear(self.embedding_dim, self.embedding_dim)
        # MIGHT be different
        # https://github.com/timbmg/Sentence-VAE/blob/master/model.py
        self.latent2hidden = Linear(self.embedding_dim, self.embedding_dim) 

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):

        x = self.encoder(x)

        # Reparameterize 
        mu = self.hidden2mean(x)
        log_var = self.hidden2logv(x)
        z = self.reparameterize(mu, log_var)

        # DECODER
        hidden = self.latent2hidden(z)
        x = self.decoder(hidden)

        return x

if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data)*0.7)

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
                    pad_packed_sequence(x, total_length=300, padding_value=-1)[0].view(-1)
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
                pad_packed_sequence(x, total_length=300, padding_value=-1)[0].view(-1)
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
        json.dump({
            "training_loss": training_loss,
            "validation loss": validation_loss}, f)

    torch.save(net.state_dict(), model_directory / f"{time_string}_state_dict.pt")
