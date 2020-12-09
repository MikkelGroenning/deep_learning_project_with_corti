from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from src.data.common import get_loader
from src.data.characters import alphabet
from src.models.character_models import (
    character_rae as rae,
    character_vrae as vrae,
    character_vrae_iaf as vrae_iaf,
    test_data,
)

from src.models.common import VariationalInference, get_trained_model

rae, t_info_rae = get_trained_model(
    rae, training_info=True, model_name="CharacterRAE"
)
vrae, t_info_rae = get_trained_model(
    vrae, training_info=True, model_name="CharacterVRAE"
)
vrae_iaf, t_info_rae = get_trained_model(
    vrae_iaf, training_info=True, model_name="CharacterVRAEIAF"
)

num_samples = 10
test_loader = get_loader(test_data, batch_size=num_samples)

x_test = next(iter(test_loader))

output_rae = rae(x_test)
output_vrae = vrae(x_test)
output_iaf = vrae_iaf(x_test)

# Sample from observation model and pack, then pad
sample_rae_packed = output_rae
sample_vrae_packed = PackedSequence(output_vrae['px'].sample(), x_test.batch_sizes)
sample_iaf_packed = PackedSequence(output_iaf['px'].sample(), x_test.batch_sizes)

sample_rae_padded, sequence_lengths = pad_packed_sequence(sample_rae_packed)
sample_vrae_padded, sequence_lengths = pad_packed_sequence(sample_vrae_packed)
sample_iaf_padded, sequence_lengths = pad_packed_sequence(sample_iaf_packed)

target_padded, _ = pad_packed_sequence(x_test)

slc = slice(None, 100)

for i, length in zip(range(num_samples), sequence_lengths):

    decoded_tweet_rae = sample_rae_padded.argmax(dim=2)[:, i]
    decoded_tweet_vrae = sample_vrae_padded[:, i]
    decoded_tweet_iaf = sample_iaf_padded[:, i]

    target_tweet = target_padded[:, i]

    print("   RAE: " + "".join(alphabet[j] for j in decoded_tweet_rae[:length])[slc])
    print("  VRAE: " + "".join(alphabet[j] for j in decoded_tweet_vrae[:length])[slc])
    print("   IAF: " + "".join(alphabet[j] for j in decoded_tweet_iaf[:length])[slc])
    print("TARGET: " + "".join(alphabet[j] for j in target_tweet[:length])[slc])

    print("")

