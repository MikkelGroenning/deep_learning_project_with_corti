import pickle
from torch.nn.utils.rnn import pad_packed_sequence

from src.models.iaf_chars import IAFChars
# from src.models.iaf_words import IAFWords
from src.models.rae_chars import RAEChars
# from src.models.rae_words import RAEWords
from src.models.vrae_chars import VRAEChars
# from src.models.vrae_words import VRAEWords

from src.models.common import *
from src.data.words import *
from src.data.characters import *

import torch

rae_c, rae_c_tinfo = get_trained_model(RAEChars, training_info=True)
vrae_c, rae_c_tinfo = get_trained_model(VRAEChars, training_info=True)
iaf_c, iaf_c_tinfo = get_trained_model(IAFChars, training_info=True)

data_words = torch.load("data/processed/200316_embedding.pkl")
with open("data/interim/hydrated/200316.pkl", "rb") as f:
    data_chars = pickle.load(f)

num_samples = 10

ds_chars = TwitterDataChars(data_chars.iloc[-10:, :].copy())
ds_words = TwitterDataWords(data_words[-10:])

words_loader = get_loader(ds_words, batch_size=num_samples)
chars_loader = get_loader(ds_chars, batch_size=num_samples)

x_words = next(iter(words_loader))
x_chars = next(iter(chars_loader))

output_rae_chars = rae_c(x_chars)
output_vrae_chars = vrae_c(x_chars)
output_iaf_chars = iaf_c(x_chars)

# Sample from observation model and pack, then pad
sample_rae_packed_chars = output_rae_chars
sample_vrae_packed_chars = PackedSequence(output_vrae_chars['px'].sample(), x_chars.batch_sizes)
sample_iaf_packed_chars = PackedSequence(output_iaf_chars['px'].sample(), x_chars.batch_sizes)

sample_rae_padded_chars, sequence_lengths = pad_packed_sequence(sample_rae_packed_chars)
sample_vrae_padded_chars, sequence_lengths = pad_packed_sequence(sample_vrae_packed_chars)
sample_iaf_padded_chars, sequence_lengths = pad_packed_sequence(sample_iaf_packed_chars)

target_padded_chars, _ = pad_packed_sequence(x_chars)

slc = slice(None, 100)

for i, length in zip(range(num_samples), sequence_lengths):

    decoded_tweet_rae = sample_rae_padded_chars.argmax(dim=2)[:, i]
    decoded_tweet_vrae = sample_vrae_padded_chars[:, i]
    decoded_tweet_iaf = sample_iaf_padded_chars[:, i]

    target_tweet = target_padded_chars[:, i]

    print("   RAE: " + "".join(alphabet[j] for j in decoded_tweet_rae[:length])[slc])
    print("  VRAE: " + "".join(alphabet[j] for j in decoded_tweet_vrae[:length])[slc])
    print("   IAF: " + "".join(alphabet[j] for j in decoded_tweet_iaf[:length])[slc])
    print("TARGET: " + "".join(alphabet[j] for j in target_tweet[:length])[slc])

    print("")
