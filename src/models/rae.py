from itertools import chain
import numpy as np
import torch
from src.data.words import TwitterDataWords

from src.models.common import (
    CriterionTrainer, Decoder, EmbeddingPacked, Encoder, ModelTrainer,
    simple_elementwise_apply,
)

from torch.nn import LSTM, Module, MSELoss, Linear
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam


class RAE(Module):

    def __init__(
        self, 
        input_dim,
        latent_features,
        encoder_hidden_size,
        decoder_hidden_size,
        output_dim=None,
    ):

        super(RAE, self).__init__()

        self.input_dim = input_dim
        self.latent_features = latent_features
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # Ensure flexibility with embedded input/output 
        if output_dim is None:
            self.output_dim=input_dim
        else:
            self.output_dim=output_dim

        self.encoder = Encoder(
            input_dim=self.input_dim,
            latent_features=self.latent_features,
            hidden_size=self.encoder_hidden_size,
        )

        self.decoder = Decoder(
            latent_features=self.latent_features,
            hidden_size=self.decoder_hidden_size, 
            output_dim=self.output_dim,
        )  

    def forward(self, x):

        batch_sizes = x.batch_sizes

        x = self.encoder(x)
        x = self.decoder(x, batch_sizes=batch_sizes)

        return x


class RAEWithEmbedder(RAE):
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



# emdedding_dim = 300

# # Default, should probably be explicit
# model_parameters = {
#     "input_dim" : emdedding_dim,
#     "latent_features" : 64,
#     "encoder_hidden_size" : 64,
#     "decoder_hidden_size" : 64,
# }

# # Training parameters

# batch_size = 2000
# max_epochs = 500

# optimizer_parameters = {"lr": 0.001}

# if __name__ == "__main__":

#     print("Loading dataset...")
#     data = torch.load('data/processed/200316_embedding.pkl')

#     split_idx = int(len(data) * 0.7)

#     dataset_train = TwitterDataWords(data[:split_idx])
#     dataset_validation = TwitterDataWords(data[split_idx:])

#     # dataset_train = TwitterDataWords(data[:1000])
#     # dataset_validation = TwitterDataWords(data[1000:1500])

#     criterion = MSELoss(reduction='sum')
#     model = RAEWords(**model_parameters)
#     optimizer = Adam(model.parameters(), **optimizer_parameters)

#     mt = CriterionTrainer(
#         criterion=criterion,
#         model=model,
#         optimizer=optimizer,
#         batch_size=batch_size,
#         max_epochs=max_epochs,
#         training_data=dataset_train,
#         validation_data=dataset_validation,
#     )

#     mt.restore_checkpoint()
#     mt.train()
