import numpy as np
import seaborn as sns
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path

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

sns.set_style("whitegrid")

overleaf_directory = Path( 'overleaf' ) / "poster" 

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

elbo = {
    "character_vrae": [],
    "character_vrae_iaf": [],
    "word_vrae": [],
    "word_vrae_iaf": [],
}

for x in tqdm(character_loader):

    loss_packed_rae = ce_loss(character_rae(x).data, x.data)
    loss_packed_vrae, diag_vrae, _ = vi(character_vrae, x)
    loss_packed_vrae_iaf, diag_vrae_iaf, _ = vi(character_vrae_iaf, x)

    ll_in_distribution["character_rae"].append(
        -float(loss_packed_rae) / len(x.data),
    )
    ll_in_distribution["character_vrae"].append(
        -float(loss_packed_vrae) / len(x.data),
    )
    ll_in_distribution["character_vrae_iaf"].append(
        -float(loss_packed_vrae_iaf) / len(x.data)
    )

    elbo["character_vrae"]
    elbo["character_vrae_iaf"]

for x in tqdm(character_loader_trump):

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


for x in tqdm(word_loader):

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

for x in tqdm(word_loader_trump):

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
    
torch.save([ll_in_distribution, ll_out_of_distribution], "ood_results.pkl")

ll_in_distribution, ll_out_of_distribution = torch.load("./ood_results.pkl")


fig, _ = plt.subplots(ncols=3, nrows=2, figsize=(14, 6))


for i, ((model, ood_dist), (_, id_dist)) in enumerate(
    zip(
        ll_out_of_distribution.items(),
        ll_in_distribution.items(),
    )
):
    

    # Only considering obervations within a reasonable intervand
    min_ood, max_ood = np.quantile(ood_dist,(0.005, 0.999)) 
    min_id, max_id = np.quantile(id_dist,(0.005, 0.999)) 

    ood_filtered = [x for x in ood_dist if x < max_ood and x > min_ood]
    id_filtered = [x for x in id_dist if x < max_id and x > min_id]

    sns.distplot(id_filtered, label="Covid", ax=fig.axes[i], kde=False,
        norm_hist=True)
    sns.distplot(ood_filtered, label="Trump", ax=fig.axes[i], kde=False,
        norm_hist=True)
    fig.axes[i].set_title(model)
    

plt.legend()


fig.tight_layout()
fig.savefig(overleaf_directory /"figures" / "ood_detection.pdf") 

