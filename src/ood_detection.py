import seaborn as sns
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.common import get_loader
from src.models.character_models import (
    CrossEntropyLoss,
    character_rae,
    character_vrae,
    character_vrae_iaf,
)
from src.models.character_models import test_data as test_data_characters
from src.models.character_models import trump_data as trump_data_characters
from src.models.common import VariationalInference, get_trained_model

from src.models.word_models import test_data as test_data_words
from src.models.word_models import trump_data as trump_data_words
from src.models.word_models import word_rae, word_vrae, word_vrae_iaf, MSELoss

character_rae = get_trained_model(character_rae, model_name="CharacterRAE")
character_vrae = get_trained_model(character_vrae, model_name="CharacterVRAE")
character_vrae_iaf = get_trained_model(character_vrae_iaf, model_name="CharacterVRAEIAF")

word_rae = get_trained_model(word_rae, model_name="WordRAE")
word_vrae = get_trained_model(word_vrae, model_name="WordVRAE")
word_vrae_iaf = get_trained_model(word_vrae_iaf, model_name="WordVRAEIAF")

character_loader = get_loader(test_data_characters, 1)
word_loader = get_loader(test_data_words, 1)

character_loader_trump = get_loader(trump_data_characters, 1)
word_loader_trump = get_loader(trump_data_words, 1)

mse_loss = MSELoss(reduction="sum")
ce_loss = CrossEntropyLoss(reduction="sum")
vi = VariationalInference()

character_rae.eval()
character_vrae.eval()
character_vrae_iaf.eval()

ll_in_distribution = {
    "character_rae": [],
    "character_vrae": [],
    "character_vrae_iaf": [],
    "word_rae": [],
    "word_vrae": [],
    "word_vrae_iaf": [],
}

ll_out_of_distribution = {
    "character_rae": [],
    "character_vrae": [],
    "character_vrae_iaf": [],
    "word_rae": [],
    "word_vrae": [],
    "word_vrae_iaf": [],
}

for x, _ in zip(tqdm(character_loader), range(1000)):

    loss_packed_rae = ce_loss(character_rae(x).data, x.data)
    loss_packed_vrae, _, _ = vi(character_vrae, x)
    loss_packed_vrae_iaf, diagnostics, _ = vi(character_vrae_iaf, x)

    ll_in_distribution["character_rae"].append(
        -float(loss_packed_rae) / len(x.data),
    )
    ll_in_distribution["character_vrae"].append(
        -float(loss_packed_vrae) / len(x.data),
    )
    ll_in_distribution["character_vrae_iaf"].append(
        -float(loss_packed_vrae_iaf) / len(x.data)
    )

for x, _ in zip(tqdm(character_loader_trump), range(1000)):

    loss_packed_rae = ce_loss(character_rae(x).data, x.data)
    loss_packed_vrae, _, _ = vi(character_vrae, x)
    loss_packed_vrae_iaf, diagnostics, _ = vi(character_vrae_iaf, x)

    ll_out_of_distribution["character_rae"].append(
        -float(loss_packed_rae) / len(x.data),
    )
    ll_out_of_distribution["character_vrae"].append(
        -float(loss_packed_vrae) / len(x.data),
    )
    ll_out_of_distribution["character_vrae_iaf"].append(
        -float(loss_packed_vrae_iaf) / len(x.data)
    )


for x, _ in zip(tqdm(word_loader), range(1000)):

    loss_packed_rae = mse_loss(word_rae(x).data, x.data)
    loss_packed_vrae, _, _ = vi(word_vrae, x)
    loss_packed_vrae_iaf, diagnostics, _ = vi(word_vrae_iaf, x)

    ll_in_distribution["word_rae"].append(
        -float(loss_packed_rae) / len(x.data),
    )
    ll_in_distribution["word_vrae"].append(
        -float(loss_packed_vrae) / len(x.data),
    )
    ll_in_distribution["word_vrae_iaf"].append(
        -float(loss_packed_vrae_iaf) / len(x.data)
    )


for x, _ in zip(tqdm(word_loader_trump), range(1000)):

    loss_packed_rae = mse_loss(word_rae(x).data, x.data)
    loss_packed_vrae, _, _ = vi(word_vrae, x)
    loss_packed_vrae_iaf, diagnostics, _ = vi(word_vrae_iaf, x)

    ll_out_of_distribution["word_rae"].append(
        -float(loss_packed_rae) / len(x.data),
    )
    ll_out_of_distribution["word_vrae"].append(
        -float(loss_packed_vrae) / len(x.data),
    )
    ll_out_of_distribution["word_vrae_iaf"].append(
        -float(loss_packed_vrae_iaf) / len(x.data)
    )

fig, _ = plt.subplots(ncols=3, nrows=2, figsize=(14, 8))

for i, ((model, ood_dist), (_, id_dist)) in enumerate(
    zip(
        ll_out_of_distribution.items(),
        ll_in_distribution.items(),
    )
):
    sns.distplot(ood_dist, label="Trump", ax=fig.axes[i], kde=False)
    sns.distplot(id_dist, label="IId", ax=fig.axes[i], kde=False)
    fig.axes[i].set_title(model)

fig.savefig("test.pdf")