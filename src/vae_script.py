
# %%
from src.models.vae import *
from src.data.data_loader import (
    get_loader, 
    TwitterDataset, 
    character_set, 
    character_to_number
)

from src.models.common import (
    EmbeddingPacked,
    get_numpy,
    get_variable,
    simple_elementwise_apply,
    cuda,
)

from torch.nn.utils.rnn import pack_padded_sequence


# %%
enc = Encoder()

# %%
data = pd.read_pickle("data/interim/hydrated/200316.pkl")
train_set = TwitterDataset(data.iloc[0:3000, :].copy())
train_loader = get_loader(train_set, 100)
validation_set = TwitterDataset(data.iloc[3000:4000, :].copy())
validation_loader = get_loader(validation_set, 100)

# %%
rvae = RecurrentVariationalAutoencoder()
vi = VariationalInference()

optimizer = Adam(rvae.parameters(), lr=0.001)

# %%
from tqdm import tqdm

training_loss = []

for i in range(50):

    # Track loss
    epoch_training_loss = 0

    rvae.train()

    # For each sentence in training set
    for x in tqdm(train_loader):

        x = get_variable(x)

        # Forward pass
        loss, diagnostics, outputs = vi(rvae, x)

        if loss.isnan():
            raise ValueError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_training_loss += get_numpy(loss.detach())

    # Save loss for plot
    training_loss.append(epoch_training_loss / len(train_set))


# %%
training_loss


# %%
import matplotlib.pyplot as plt 


# %%



# %%
64039280043535 / 300_000


# %%
loss, diagnostics, outputs = vi(rvae, x)


# %%
loss


# %%
x


# %%



