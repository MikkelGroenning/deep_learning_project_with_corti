import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from src.data.data_loader import TwitterDataset, alphabet, get_loader
from src.models.common import (EmbeddingPacked, get_numpy, get_variable,
                               simple_elementwise_apply)
from torch.nn import LSTM, CrossEntropyLoss, Linear, Module, ReLU, Sequential
from torch.nn.utils.rnn import pack_sequence
from torch.optim import SGD


# Custom dataset for predictive RNN modelling
class PredictiveDataset(TwitterDataset):
    def __getitem__(self, indices):

        values = sorted([self.encoded[i] for i in indices], key=lambda x: -len(x))

        inputs = [value[:-1] for value in values]
        targets = [value[1:] for value in values]

        X = pack_sequence(sorted(inputs, key=lambda x: -len(x)))
        y = pack_sequence(sorted(targets, key=lambda x: -len(x)))

        return X, y


# Model defition
class SimpleRNN(Module):
    def __init__(self):

        super().__init__()

        self.embedding = EmbeddingPacked(
            num_embeddings=len(alphabet),
            embedding_dim=10,
        )

        self.lstm = LSTM(
            input_size=10,
            hidden_size=50,
            num_layers=1,
            bidirectional=False,
        )

        self.dense = Sequential(
            Linear(in_features=50, out_features=100),
            ReLU(inplace=True),
            Linear(in_features=100, out_features=len(alphabet)),
        )

    def forward(self, x):

        # Define packing here?
        x = self.embedding(x)
        packed_output, (ht, ct) = self.lstm(x)

        return simple_elementwise_apply(self.dense, packed_output)


if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data)*0.7)

    dataset_train = PredictiveDataset(data.iloc[:split_idx, :].copy())
    dataset_validation = PredictiveDataset(data.iloc[split_idx:, :].copy())

    cuda = torch.cuda.is_available()
    if cuda:
        print("Using CUDA...")
    else:
        print("Using CPU...")

    batch_size = 5000
    train_loader = get_loader(dataset_train, batch_size, pin_memory=cuda)
    validation_loader = get_loader(dataset_validation, batch_size, pin_memory=cuda)

    net = SimpleRNN()

    if cuda:
        net = net.cuda()

    # Hyper-parameters
    num_epochs = 2

    # Define a loss function and optimizer for this problem
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

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
            for X, y in validation_loader:
                
                # One-hot encode input and target sequence
                X = get_variable(X)
                y = get_variable(y)
            
                # Forward pass
                packed_outputs = net(X)
                # Backward pass
                loss = criterion(packed_outputs.data, y.data)  
                # Update loss
                epoch_validation_loss += get_numpy(loss.detach())
            

        net.train()

        # For each sentence in training set
        for X, y in train_loader:

            X = get_variable(X)
            y = get_variable(y)

            # Forward pass
            packed_outputs = net(X)

            # Compute loss
            loss = criterion(packed_outputs.data, y.data)

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
