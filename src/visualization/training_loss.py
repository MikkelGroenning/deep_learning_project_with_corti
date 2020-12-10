from pathlib import Path
import seaborn as sns
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

figure_directory = Path(__file__).parent.parent.parent / 'reports' / 'figures'
overleaf_directory = Path(__file__).parent.parent.parent / 'overleaf' / "poster"

sns.set_style("whitegrid")

character_rae, t_info_rae_c = get_trained_model(character_rae, training_info=True, model_name="CharacterRAE")
character_vrae, t_info_vrae_c = get_trained_model(character_vrae, training_info=True, model_name="CharacterVRAE")
character_vrae_iaf, t_info_vrae_iaf_c = get_trained_model(character_vrae_iaf, training_info=True, model_name="CharacterVRAEIAF")

word_rae, t_info_rae_w = get_trained_model(word_rae, training_info=True, model_name="WordRAE")
word_vrae, t_info_vrae_w = get_trained_model(word_vrae, training_info=True, model_name="WordVRAE")
word_vrae_iaf, t_info_vrae_iaf_w = get_trained_model(word_vrae_iaf, training_info=True, model_name="WordVRAEIAF")

fig, _ =  plt.subplots( nrows=2, ncols=3, figsize=(14,5))

for i, (t_info, model_desc) in enumerate([
    (t_info_rae_c, "Character-based RAE"),
    (t_info_vrae_c, "Character-based VRAE"),
    (t_info_vrae_iaf_c, "Character-based VRAE with IAF"),
    (t_info_rae_w, "Word-based RAE"),
    (t_info_vrae_w, "Word-based VRAE"),
    (t_info_vrae_iaf_w, "Word-based VRAE with IAF"),
    ]):
    lines = [
    fig.axes[i].plot(t_info["training_loss"]),
    fig.axes[i].plot(t_info["validation_loss"]),
        ]
    fig.axes[i].set_xlabel("epoch")
    fig.axes[i].set_title(model_desc)

    if i == 0:
        fig.axes[i].set_ylabel("loss (NLL)")
        fig.axes[i].set_ylim(330, 370)
    if i in (1, 2):
        fig.axes[i].set_ylabel("loss (-ELBO(beta=0.1))")
        fig.axes[i].set_ylim(330, 370)
    if i == 3:
        fig.axes[i].set_ylim(35, 70)
        fig.axes[i].set_ylabel("loss (MSE)")
    if i in (4, 5):
        fig.axes[i].set_ylabel("loss (-ELBO(beta=0.1))")
        fig.axes[i].set_ylim(-10000, 10000)

leg1 = mpatches.Patch(color='C0')
leg2 = mpatches.Patch(color='C1')
fig.legend(handles=[leg1, leg2], labels=[
    "Training loss", "Validation loss"], loc="lower center", ncol=2)

fig.tight_layout()
fig.savefig(overleaf_directory /"figures" / "model_convergence.pdf") 

