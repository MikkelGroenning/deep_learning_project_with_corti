import seaborn as sns
import torch
from tqdm import tqdm

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

sns.set_style("whitegrid")

if __name__ == "__main__":

    character_rae = get_trained_model(
        character_rae, model_name="CharacterRAE", subdir="beta-scheduling", latest=True
    )
    character_vrae = get_trained_model(
        character_vrae, model_name="CharacterVRAE", subdir="beta-scheduling", latest=True
    )
    character_vrae_iaf = get_trained_model(
        character_vrae_iaf, model_name="CharacterVRAEIAF", subdir="beta-scheduling", latest=True
    )

    character_loader = get_loader(test_data_characters, 1)

    character_loader_trump = get_loader(trump_data_characters, 1)

    ce_loss = CrossEntropyLoss(reduction="sum")
    vi = VariationalInference()

    character_rae.eval()
    character_vrae.eval()
    character_vrae_iaf.eval()

    ll_in_distribution = {
        "character_rae": [],
        "character_vrae": [],
        "character_vrae_iaf": [],
    }


    ll_out_of_distribution = {
        "character_rae": [],
        "character_vrae": [],
        "character_vrae_iaf": [],
    }

    diagnostics_in_distribution = {
        "character_vrae": {
            "elbo": [],
            "log_px": [],
            "kl": [],
        },
        "character_vrae_iaf": {
            "elbo": [],
            "log_px": [],
            "kl": [],
        },
    }

    diagnostics_out_of_distribution = {
        "character_vrae": {
            "elbo": [],
            "log_px": [],
            "kl": [],
        },
        "character_vrae_iaf": {
            "elbo": [],
            "log_px": [],
            "kl": [],
        },
    }


    for x in tqdm(character_loader):
    # for x, _ in zip(tqdm(character_loader), range(100)):

        loss_packed_rae = ce_loss(character_rae(x).data, x.data)
        loss_packed_vrae, diag_vrae, _ = vi(character_vrae, x)
        loss_packed_vrae_iaf, diag_vrae_iaf, _ = vi(character_vrae_iaf, x)

        ll_in_distribution["character_rae"].append(
            -float(loss_packed_rae) / len(x.data),
        )
        ll_in_distribution["character_vrae"].append(
            float(diag_vrae["elbo"]) / len(x.data),
        )
        ll_in_distribution["character_vrae_iaf"].append(
            float(diag_vrae_iaf["elbo"]) / len(x.data),
        )

        diagnostics_in_distribution["character_vrae"]["elbo"].append(
            float(diag_vrae["elbo"])
        )
        diagnostics_in_distribution["character_vrae"]["log_px"].append(
            float(diag_vrae["log_px"])
        )
        diagnostics_in_distribution["character_vrae"]["kl"].append(float(diag_vrae["kl"]))

        diagnostics_in_distribution["character_vrae_iaf"]["elbo"].append(
            float(diag_vrae_iaf["elbo"])
        )
        diagnostics_in_distribution["character_vrae_iaf"]["log_px"].append(
            float(diag_vrae_iaf["log_px"])
        )
        diagnostics_in_distribution["character_vrae_iaf"]["kl"].append(
            float(diag_vrae_iaf["kl"])
        )

    for x in tqdm(character_loader_trump):
    # for x, _ in zip(tqdm(character_loader_trump), range(100)):

        loss_packed_rae = ce_loss(character_rae(x).data, x.data)
        loss_packed_vrae, diag_vrae, _ = vi(character_vrae, x)
        loss_packed_vrae_iaf, diag_vrae_iaf, _ = vi(character_vrae_iaf, x)

        ll_out_of_distribution["character_rae"].append(
            -float(loss_packed_rae) / len(x.data),
        )
        ll_out_of_distribution["character_vrae"].append(
            float(diag_vrae["elbo"]) / len(x.data),
        )
        ll_out_of_distribution["character_vrae_iaf"].append(
            float(diag_vrae_iaf["elbo"]) / len(x.data),
        )

        diagnostics_out_of_distribution["character_vrae"]["elbo"].append(
            float(diag_vrae["elbo"])
        )
        diagnostics_out_of_distribution["character_vrae"]["log_px"].append(
            float(diag_vrae["log_px"])
        )
        diagnostics_out_of_distribution["character_vrae"]["kl"].append(
            float(diag_vrae["kl"])
        )

        diagnostics_out_of_distribution["character_vrae_iaf"]["elbo"].append(
            float(diag_vrae_iaf["elbo"])
        )
        diagnostics_out_of_distribution["character_vrae_iaf"]["log_px"].append(
            float(diag_vrae_iaf["log_px"])
        )
        diagnostics_out_of_distribution["character_vrae_iaf"]["kl"].append(
            float(diag_vrae_iaf["kl"])
        )

    torch.save(
        {
            "ll_in_distribution": ll_in_distribution,
            "ll_out_of_distribution": ll_out_of_distribution,
            "diagnostics_in_distribution": diagnostics_in_distribution,
            "diagnostics_out_of_distribution": diagnostics_out_of_distribution,
        },
        Path(".") / "reports" / "new_ood_results.pkl",
    )
