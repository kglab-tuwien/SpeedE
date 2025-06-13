import re
from typing import ClassVar, Mapping, Any
import torch
from pykeen.models import ERModel
from pykeen.losses import NSSALoss

from Interactions import ExpressivE_Interaction, SpeedE_Interaction


class SpeedE(ERModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=dict(type=int, low=9, high=11, scale='power', base=2),
        p=dict(type=int, low=1, high=2),
        min_denom=dict(type=float, low=2e-1, high=8e-1, step=1e-1),
    )

    loss_default = NSSALoss
    loss_default_kwargs = dict(
        margin=3.0,
        adversarial_temperature=2.0,
        reduction="sum",
    )

    def __init__(
        self,
        embedding_dim: int = 50,
        p: int = 2,
        min_denom: float = 0.5,
        tanh_map: bool = True,
        interactionMode: str = "baseExpressivE",
        **kwargs,
    ) -> None:
        if interactionMode == "baseExpressivE":
            # <<<<<< ExpressivE >>>>>>
            super().__init__(
                interaction=ExpressivE_Interaction(
                    p=p,
                    min_denom=min_denom,
                    tanh_map=tanh_map,
                ),
                entity_representations_kwargs=dict(
                    shape=(embedding_dim,),
                ),
                relation_representations_kwargs=dict(
                    shape=(6 * embedding_dim,),
                ),  # d_h, d_t, c_h, c_t, s_h, s_t
                **kwargs,
            )

        elif re.search(r"SpeedE", interactionMode):
            if min_denom > 0:
                raise ValueError(
                    f"{interactionMode} SpeedE does not use the <min_denom> argument. "
                    "Please set <min_denom>=0."
                )

            # parse number of inequations, mode suffix, and fixed_d
            number_inequations = int(re.search(r"SpeedE_(\d+)", interactionMode)[1])

            # base parameters
            n_paras = 2
            n_single_paras = 0
            mode = "Min_SpeedE"

            if re.search(r"_Eq", interactionMode):
                mode = "Eq_SpeedE"
                n_single_paras += 1
            elif re.search(r"_Diff", interactionMode):
                mode = "Diff_SpeedE"
                n_single_paras += 2

            m = re.search(r"_dIs(\d+\.?\d*)", interactionMode)
            if not m:
                raise ValueError("No fixed_d specified in interactionMode (use '_dIs<value>').")
            fixed_d = float(m[1])

            super().__init__(
                interaction=SpeedE_Interaction(
                    p=p,
                    tanh_map=tanh_map,
                    number_inequations=number_inequations,
                    mode=mode,
                    fixed_d=fixed_d,
                ),
                entity_representations_kwargs=dict(
                    shape=(embedding_dim,),
                ),
                relation_representations_kwargs=dict(
                    shape=(n_paras * number_inequations * embedding_dim + n_single_paras,),
                ),
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown interactionMode: {interactionMode}")
