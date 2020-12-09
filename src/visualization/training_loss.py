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

character_rae, t_info_rae_c = get_trained_model(character_rae, training_info=True, model_name="CharacterRAE")
character_vrae, t_info_vrae_c = get_trained_model(character_vrae, training_info=True, model_name="CharacterVRAE")
character_vrae_iaf, t_info_vrae_iaf_c = get_trained_model(character_vrae_iaf, training_info=True, model_name="CharacterVRAEIAF")

word_rae, t_info_rae_w = get_trained_model(word_rae, training_info=True, model_name="WordRAE")
word_vrae, t_info_vrae_w = get_trained_model(word_vrae, training_info=True, model_name="WordVRAE")
word_vrae_iaf, t_info_vrae_iaf_w = get_trained_model(word_vrae_iaf, training_info=True, model_name="WordVRAEIAF")

fig, _ =  plt.subplots( nrows=2, ncols=3 )

for i, t_info in enumerate([t_info_rae_c, t_info_vrae_c, t_info_vrae_iaf_c,
    t_info_rae_w, t_info_vrae_w, t_info_vrae_iaf_w ]):

    fig.axes[i].plot(t_info["training_loss"])
