from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from src.data.common import get_loader
from src.data.characters import alphabet
from src.models.character_models import (
    character_rae,
    character_vrae,
    character_vrae_iaf,
    test_data as test_data_characters,
    CrossEntropyLoss
)
from src.models.word_models import (
    word_rae,
    word_vrae,
    word_vrae_iaf,
    test_data as test_data_words,
    MSELoss
)


from src.models.common import VariationalInference, get_trained_model

import seaborn as sns
from tqdm import tqdm

character_loader = get_loader(test_data_characters, 1)
word_loader = get_loader(test_data_words, 1)

mse_loss = MSELoss(reduction="sum")
ce_loss = CrossEntropyLoss(reduction="sum")
vi = VariationalInference()

character_rae.eval()
character_vrae.eval()
character_vrae_iaf.eval()

for x, _ in zip(character_loader, range(10_000)):

    loss_packed_rae = ce_loss(character_rae(x).data, x.data)
    loss_packed_vrae, _, _ = vi(character_vrae, x)
    loss_packed_vrae_iaf, diagnostics, _ = vi(character_vrae_iaf, x)

    print(
    -float(loss_packed_rae)/len(x.data),
    -float(loss_packed_vrae)/len(x.data),
    -float(loss_packed_vrae_iaf)/len(x.data)
    )