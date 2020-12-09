# %%

from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
import random

from src.models.toy_models import rae, vrae, test_data
from src.models.toy_models import vrae_iaf as iaf
from src.models.common import get_trained_model, get_loader

import matplotlib.pyplot as plt
import seaborn as sns

figure_directory = Path(__file__).parent.parent.parent / 'reports' / 'figures'
overleaf_directory = Path(__file__).parent.parent.parent / 'overleaf' / "poster"


sns.set_style("whitegrid")

random.seed(44)

n_test = 1000
x_test = next(iter(get_loader(test_data, n_test)))
target_padded, sequence_lengths = pad_packed_sequence(x_test)
target_averages = []
for i, length in enumerate(sequence_lengths):
    arr = target_padded[:,i][:length-1].numpy()
    target_averages.append(arr.mean())
    
batch_sizes = x_test.batch_sizes

num_emph = 4
emph_index = sorted(random.choices(range(len(test_data)), k=num_emph))
emph_packed = pack_padded_sequence( target_padded[:,emph_index,:], sequence_lengths[emph_index])


rae, t_info_rae = get_trained_model(rae, model_name="ToyRAE", training_info=True)
vrae, t_info_vrae = get_trained_model(vrae, model_name="ToyVRAE", training_info=True)
iaf, t_info_iaf = get_trained_model(iaf, model_name="ToyVRAEIAF", training_info=True)

fig, axes =  plt.subplots(nrows=3, ncols=3, figsize=(14,8))

# %% Recurrent Autoencoder
plt.figure()
plt.plot(t_info_rae["training_loss"])
plt.plot(t_info_rae["validation_loss"])
plt.savefig(figure_directory / "rae_toy_loss.pdf")

#%%
output_rae = rae(x_test)
latent_rae = rae.encoder(x_test)

#%%
sns.scatterplot(
    x=latent_rae[:,0].detach().numpy(),
    y=latent_rae[:,1].detach().numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    ax=axes[0][0],
    )

axes[0][0].get_legend().remove()
# %%
sns.scatterplot(
    x=latent_rae[:,0].detach().numpy(),
    y=latent_rae[:,1].detach().numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    alpha=0.2,
    ax=axes[0][1],
    )
sns.scatterplot(
    x=latent_rae[emph_index,0].detach().numpy(),
    y=latent_rae[emph_index,1].detach().numpy(),
    c=[f"C{k}" for k in range(num_emph)],
    ax=axes[0][1],
)
axes[0][1].get_legend().remove()
# %%
reconstrued_rae_padded, _ = pad_packed_sequence(output_rae)
for k, i in enumerate(emph_index):

    decoded_rae = reconstrued_rae_padded[:, i][:sequence_lengths[i]]
    target = target_padded[:, i][:sequence_lengths[i]]
    
    axes[0][2].plot(decoded_rae.detach().numpy(), color = f'C{k}', linestyle='dashed')
    axes[0][2].plot(target.detach().numpy(), color = f'C{k}')

# plt.savefig(figure_directory / "rae_toy_reconstruction.pdf")

# %% Variational Recurrent Autoencoder
plt.figure()
plt.plot(t_info_vrae["training_loss"])
plt.plot(t_info_vrae["validation_loss"])

plt.ylim(top=10, bottom=-20)
plt.savefig(figure_directory / "vrae_toy_loss.pdf")

# %%
output_vrae = vrae(x_test)
output_vrae_emph = vrae(emph_packed)
latent_sample_vrae = output_vrae['qz'].sample()
latent_emph_samples_vrae = torch.stack([output_vrae_emph['qz'].sample() for _ in range(10_000)])
observation_sample_vrae = output_vrae['px'].sample()

# %%
# plt.figure()
sns.scatterplot(
    x=latent_sample_vrae[:,0].numpy(),
    y=latent_sample_vrae[:,1].numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    ax=axes[1][0],
)
axes[1][0].get_legend().remove()
# plt.savefig(figure_directory / "vrae_toy_latent.pdf")


# %%
# plt.figure()
sns.scatterplot(
    x=latent_sample_vrae[:,0].detach().numpy(),
    y=latent_sample_vrae[:,1].detach().numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    alpha=0.2,
    ax=axes[1][1],
    )

for i in range(num_emph):
    sns.kdeplot(
        x=latent_emph_samples_vrae[:, i, 0],
        y=latent_emph_samples_vrae[:, i, 1],
        levels=2,
        ax=axes[1][1],
    )
    
sns.scatterplot(
    x=latent_sample_vrae[emph_index,0].detach().numpy(),
    y=latent_sample_vrae[emph_index,1].detach().numpy(),
    c=[f"C{k}" for k in range(num_emph)],
    markers="x",
    ax=axes[1][1],
)
axes[1][1].get_legend().remove()
# plt.savefig(figure_directory / "vrae_toy_emph.pdf")

# %%
# plt.figure()
observation_sample_vrae_packed = PackedSequence(
    observation_sample_vrae,
    batch_sizes,
)
observation_sample_vrae_packed, _ = pad_packed_sequence(observation_sample_vrae_packed)

for k, i in enumerate(emph_index):

    decoded_vrae = observation_sample_vrae_packed[:, i][:sequence_lengths[i]]
    target = target_padded[:, i][:sequence_lengths[i]]
    
    axes[1][2].plot(decoded_vrae.detach().numpy(), color = f'C{k}', linestyle='dashed')
    axes[1][2].plot(target.detach().numpy(), color = f'C{k}')

# plt.savefig(figure_directory / "vrae_toy_reconstruction.pdf")
# %% IAF Variational Recurrent Autoencoder
plt.figure()

plt.plot(t_info_iaf["training_loss"])
plt.plot(t_info_iaf["validation_loss"])
plt.ylim(top=10, bottom=-20)

plt.savefig(figure_directory / "iaf_toy_loss.pdf")

# %%
output_iaf = iaf(x_test)
output_iaf_emph = iaf(emph_packed)
latent_sample_iaf = output_iaf['z'].detach()
latent_emph_samples_iaf = torch.stack([iaf(emph_packed)['z'] for _ in range(10_000)])
observation_sample_iaf = output_iaf['px'].sample()

# %%
sns.scatterplot(
    x=latent_sample_iaf[:,0].numpy(),
    y=latent_sample_iaf[:,1].numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    ax=axes[2][0],
)
axes[2][0].get_legend().remove()

sns.scatterplot(
    x=latent_sample_iaf[:,0].detach().numpy(),
    y=latent_sample_iaf[:,1].detach().numpy(),
    size=sequence_lengths.numpy(),
    hue=target_averages,
    alpha=0.2,
    ax=axes[2][1],
    )

for i in range(num_emph):
    sns.kdeplot(
        x=latent_emph_samples_iaf[:, i, 0].detach().numpy(),
        y=latent_emph_samples_iaf[:, i, 1].detach().numpy(),
        levels=2,
        ax=axes[2][1],
    )
    
sns.scatterplot(
    x=latent_sample_iaf[emph_index,0].detach().numpy(),
    y=latent_sample_iaf[emph_index,1].detach().numpy(),
    c=[f"C{k}" for k in range(num_emph)],
    markers="x",
    ax=axes[2][1],
)

axes[2][1].get_legend().remove()

observation_sample_iaf_packed = PackedSequence(
    observation_sample_iaf,
    batch_sizes,
)
observation_sample_iaf_packed, _ = pad_packed_sequence(observation_sample_iaf_packed)

for k, i in enumerate(emph_index):

    decoded_iaf = observation_sample_iaf_packed[:, i][:sequence_lengths[i]]
    target = target_padded[:, i][:sequence_lengths[i]]
    
    axes[2][2].plot(decoded_iaf.detach().numpy(), color = f'C{k}', linestyle='dashed')
    axes[2][2].plot(target.detach().numpy(), color = f'C{k}')

fig.tight_layout()
fig.savefig(overleaf_directory / "figures" / "toy_examples" / "toy_performance.pdf")
fig.savefig(figure_directory / "toy_performance.pdf")
