
import pandas as pd
from src.data.characters import TwitterDataChars, alphabet

from src.models.common import CriterionTrainer, EmbeddingPacked

from src.models.rae_words import RAE

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

num_classes = len(alphabet)

model_parameters = {
    "input_dim": num_classes,
    "embedding_dim": 10,
    "latent_features": 64,
    "encoder_hidden_size": 64,
    "decoder_hidden_size": 64,
}

# Training parameters

batch_size = 100
max_epochs = 20

optimizer_parameters = {"lr": 0.001}

if __name__ == "__main__":

    print("Loading dataset...")
    data = pd.read_pickle("data/interim/hydrated/200316.pkl")

    # split_idx = int(len(data) * 0.7)

    # dataset_train = TwitterDataChars(data.iloc[:split_idx, :].copy())
    # dataset_validation = TwitterDataChars(data.iloc[split_idx:, :].copy())

    dataset_train = TwitterDataChars(data.iloc[:1000, :].copy())
    dataset_validation = TwitterDataChars(data.iloc[1000:1500, :].copy())

    criterion = CrossEntropyLoss(reduction="sum")
    model = RAEWithEmbedder(**model_parameters)
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
