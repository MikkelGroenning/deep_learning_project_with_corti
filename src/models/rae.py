from src.models.common import Decoder, EmbeddingPacked, Encoder

from torch.nn import Module


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
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

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