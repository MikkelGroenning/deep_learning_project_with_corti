from itertools import chain

import numpy as np
import pandas as pd
from seaborn.distributions import ecdfplot
from src.data.characters import TwitterDataChars, alphabet

from src.models.common import (
    CriterionTrainer, EmbeddingPacked,
    simple_elementwise_apply,
    Encoder,
    Decoder
)

from torch.nn import LSTM, CrossEntropyLoss, Linear, Module
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam



class RAEChars(Module):

    def __init__(
        self, 
        input_dim,
        embedding_dim,
        latent_features,
        encoder_hidden_size,
        decoder_hidden_size,
        embedding=None
    ):

        super(RAEChars, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.latent_features = latent_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = Encoder(
            input_dim=self.embedding_dim,
            latent_features=self.latent_features,
            hidden_size=self.encoder_hidden_size,
        )

        self.decoder = Decoder(
            output_dim=self.input_dim,
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size, 
        )  

        if embedding is None:
            self.embedding = EmbeddingPacked(
                    num_embeddings=input_dim,
                    embedding_dim=embedding_dim,
                )
        else:
            self.embedding = embedding
            

    def forward(self, x):

        batch_sizes = x.batch_sizes

        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x, batch_sizes=batch_sizes)

        return x

# Default, should probably be explicit

num_classes = len(alphabet)

model_parameters = {
    "input_dim" : num_classes,
    "embedding_dim" : 10,
    "latent_features" : 64,
    "encoder_hidden_size" : 64,
    "decoder_hidden_size" : 64,
}

# Training parameters

batch_size = 2000
max_epochs = 500

optimizer_parameters = {"lr": 0.001}

if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    split_idx = int(len(data) * 0.7)

    dataset_train = TwitterDataChars(data.iloc[:split_idx, :].copy())
    dataset_validation = TwitterDataChars(data.iloc[split_idx:, :].copy())

    # dataset_train = TwitterDataChars(data.iloc[:1000, :].copy())
    # dataset_validation = TwitterDataChars(data.iloc[1000:1500, :].copy())

    criterion = CrossEntropyLoss(reduction="sum")
    model = RAEChars(**model_parameters)
    optimizer = Adam(model.parameters(), **optimizer_parameters)

    mt = CriterionTrainer(
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=max_epochs,
        training_data=dataset_train,
        validation_data=dataset_validation,
    )

    mt.restore_checkpoint()
    mt.train()
