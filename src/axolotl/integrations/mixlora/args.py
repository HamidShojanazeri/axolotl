# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for handling MixLoRA input arguments.
"""

from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from .constants import MIXLORA_DEFAULTS


class MixLoraArgs(BaseModel):
    """
    Input args for MixLoRA.
    """

    DEFAULT_NUM_EXPERTS: ClassVar[int] = int(MIXLORA_DEFAULTS["mixlora_num_experts"])
    DEFAULT_TOP_K: ClassVar[int] = int(MIXLORA_DEFAULTS["mixlora_top_k"])
    DEFAULT_ROUTER_AUX_LOSS_COEF: ClassVar[float] = MIXLORA_DEFAULTS[
        "mixlora_router_aux_loss_coef"
    ]
    DEFAULT_ROUTER_INIT_RANGE: ClassVar[float] = MIXLORA_DEFAULTS[
        "mixlora_router_init_range"
    ]
    DEFAULT_JITTER_NOISE: ClassVar[float] = MIXLORA_DEFAULTS["mixlora_jitter_noise"]

    mixlora_num_experts: int | None = Field(
        default=DEFAULT_NUM_EXPERTS,
        ge=1,
        json_schema_extra={
            "description": "Number of LoRA experts per FFN layer for MixLoRA (default 8)"
        },
    )
    mixlora_top_k: int | None = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        json_schema_extra={
            "description": "Number of experts to route each token to (default 2)"
        },
    )
    mixlora_router_aux_loss_coef: float | None = Field(
        default=DEFAULT_ROUTER_AUX_LOSS_COEF,
        ge=0.0,
        json_schema_extra={
            "description": "Coefficient for the auxiliary load balance loss (default 0.01)"
        },
    )
    mixlora_router_init_range: float | None = Field(
        default=DEFAULT_ROUTER_INIT_RANGE,
        gt=0.0,
        json_schema_extra={
            "description": "Initialization range for router weights (default 0.02)"
        },
    )
    mixlora_jitter_noise: float | None = Field(
        default=DEFAULT_JITTER_NOISE,
        ge=0.0,
        json_schema_extra={
            "description": "Noise added to router inputs during training for exploration (default 0.0)"
        },
    )
    mixlora_expert_lora_r: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Separate LoRA rank for MixLoRA experts. Defaults to lora_r if not set."
        },
    )
    mixlora_expert_lora_alpha: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Separate LoRA alpha for MixLoRA experts. Defaults to lora_alpha if not set."
        },
    )
    mixlora_expert_lora_dropout: float | None = Field(
        default=None,
        ge=0.0,
        json_schema_extra={
            "description": "Separate LoRA dropout for MixLoRA experts. Defaults to lora_dropout if not set."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_mixlora_compat(cls, data):
        if data.get("adapter") != "mixlora":
            return data
        if not data.get("lora_r"):
            raise ValueError("lora_r is required when using the mixlora adapter")
        if not data.get("lora_alpha"):
            raise ValueError("lora_alpha is required when using the mixlora adapter")
        if data.get("flash_attn_fuse_mlp"):
            raise ValueError("Fused modules are not supported with MixLoRA")
        if data.get("flash_attn_fuse_qkv"):
            raise ValueError("flash_attn_fuse_qkv is not supported with MixLoRA")
        if data.get("rl") is not None:
            raise ValueError(
                "MixLoRA is not compatible with RL training (DPO/KTO/GRPO). "
                "Use adapter: lora or qlora for RL fine-tuning."
            )
        return data

    @model_validator(mode="after")
    def validate_mixlora_top_k(self):
        num_experts = (
            self.mixlora_num_experts
            if self.mixlora_num_experts is not None
            else self.DEFAULT_NUM_EXPERTS
        )
        top_k = (
            self.mixlora_top_k if self.mixlora_top_k is not None else self.DEFAULT_TOP_K
        )

        if top_k > num_experts:
            raise ValueError(
                f"mixlora_top_k ({top_k}) must be <= "
                f"mixlora_num_experts ({num_experts})"
            )
        return self
