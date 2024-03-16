import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
import torch.nn as nn
from typing import Union
from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file


class MambaForNER(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        num_class: int = None,
        device=None,
        dtype=torch.float16
    ) -> None:

        """
        class MambaConfig:

            d_model: int = 2560
            n_layer: int = 64
            vocab_size: int = 50277
            ssm_cfg: dict = field(default_factory=dict)
            rms_norm: bool = True
            residual_in_fp32: bool = True
            fused_add_norm: bool = True
            pad_vocab_size_multiple: int = 8
            tie_embeddings: bool = True
        """

        super().__init__()
        self.mamba_config = config
        self.backbone = MambaLMHeadModel(self.mamba_config,
                                         initializer_cfg=initializer_cfg,
                                         device=device,
                                         dtype=dtype)

        self.backbone.lm_head = nn.Linear(in_features=self.mamba_config.d_model,
                                          out_features=num_class, bias=False, dtype=dtype)

        # weight_file = cached_file('state-spaces/mamba-1.4b', WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        state_dict = torch.load('mamba-370m.pt')
        del state_dict['lm_head.weight']
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):

        output = self.backbone(input_ids,
                               position_ids=position_ids,
                               inference_params=inference_params,
                               num_last_tokens=num_last_tokens)

        return output
