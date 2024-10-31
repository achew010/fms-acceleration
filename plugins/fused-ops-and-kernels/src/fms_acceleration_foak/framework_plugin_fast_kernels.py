# Copyright The FMS HF Tuning Authors
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

# Standard
from typing import Dict, Set, Tuple

# Third Party
from fms_acceleration import AccelerationPlugin, AccelerationPluginConfigError
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from transformers import TrainingArguments
import torch

# Local
from .framework_plugin_fast_quantized_peft import lora_adapters_switch_ddp_from_fsdp


# consider rewriting register_foak_model_patch_rules into something
# like this also
def register_foak_model_patch_rules2(base_type: str, filter_endswith: Set[str] = None):

    # Third Party
    from fms_acceleration.model_patcher import (  # pylint: disable=import-outside-toplevel
        ModelPatcher,
    )

    # Local
    from .models import (  # pylint: disable=import-outside-toplevel
        gpt_bigcode,
        granite,
        llama,
        mistral,
        mixtral,
    )

    rules = [
        *gpt_bigcode.get_mp_rules(base_type),
        *granite.get_mp_rules(base_type),
        *llama.get_mp_rules(base_type),
        *mistral.get_mp_rules(base_type),
        *mixtral.get_mp_rules(base_type),
    ]

    if filter_endswith is not None:
        # filter rules
        rules = [
            r for r in rules if any(r.rule_id.endswith(x) for x in filter_endswith)
        ]

    for _rule in rules:
        ModelPatcher.register(_rule)


# maybe this we should define envvars
FILTER_MAP = {
    "fused_lora": {"qkvo", "mlp"},
    "fast_loss": "cross-ent",
    "fast_rms_layernorm": "rms",
    "fast_rope_embeddings": "rope",
}


class FastKernelsAccelerationPlugin(AccelerationPlugin):

    # NOTE: may remove this when we have generic model rules
    restricted_model_archs = [
        "GraniteForCausalLM",
        "GPTBigCodeForCausalLM",
        "MixtralForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # NOTE: unfortunately we have to do this now, there is no good way to specify mutiple
        # keys
        try:
            self.configurations = self._check_config_and_maybe_check_values(
                key="training.fused_ops_and_kernels",
            )
        except AccelerationPluginConfigError:
            self.configurations = self._check_config_and_maybe_check_values(
                key="peft.quantization.fused_ops_and_kernels",
            )

        self.configurations["base_layer"] = self._check_config_and_maybe_check_values(
            key="base_layer", values=["auto_gptq", "bitsandbytes"], default="auto_gptq"
        )
        self.configurations["fused_lora"] = self._check_config_and_maybe_check_values(
            key="fused_lora", values=[False, True], default=True
        )
        self.configurations["fast_loss"] = self._check_config_and_maybe_check_values(
            key="fast_loss", values=[False, True], default=True
        )
        self.configurations["fast_rms_layernorm"] = (
            self._check_config_and_maybe_check_values(
                key="fast_rms_layernorm", values=[False, True], default=True
            )
        )
        self.configurations["fast_rope_embeddings"] = (
            self._check_config_and_maybe_check_values(
                key="fast_rope_embeddings", values=[False, True], default=True
            )
        )

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        has_quant = getattr(model, "quantization_method", None)

        if has_quant:
            # - only in the case where quant case, that we enforce the mixed precision settings
            # - this is mostly for the fused-loras
            assert (
                train_args.bf16 is True or train_args.fp16 is True
            ), f"{self.__class__} requires mixed precision argument `--fp16` or `--bf16`"

        # This is designed to be a passthrough if training scenario is
        # full finetuning or standard peft, fused-lora rules (only meant for qpeft)
        # will still be installed but never triggered
        # if no peft layer is detected at the point of patching

        # some logic to omit terms from the filter if logic precludes
        omitted = set()
        if has_quant is None:
            # - fused_lora only required for quant-peft
            omitted.add("fused_lora")

        terms = set()
        for k, v in self.configurations.items():
            if k in FILTER_MAP and k not in omitted:
                ts = FILTER_MAP[k]
                if isinstance(ts, str):
                    ts = {ts}
                if isinstance(v, bool) and v is False:
                    continue
                terms.update(ts)

        # wrapper function to register foak patches
        # - the base layer setting below will be ignored in non quantized-lora settings
        register_foak_model_patch_rules2(
            base_type=self.configurations["base_layer"], filter_endswith=terms
        )
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        # This callback applies only for qpeft
        # should not install this for full FT and standard peft
        is_quantized = getattr(model, "quantization_method", None)
        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
            and is_quantized is not None
        ):
            # This function installs grad reduction hooks on adapters if
            # FSDP is detected. Because of incompatibility between FSDP and
            # fused modules, adapters are not sharded - instead
            # accumulated gradients from adapters in each device are reduced
            # in these grad reduce hooks
            # This function might be removed in future if the incompatiblity
            # is resolved
            lora_adapters_switch_ddp_from_fsdp(
                [mod for mod in model.modules() if isinstance(mod, LoraLayer)],
                accelerator.state.fsdp_plugin,
            )
        return callbacks


# register
AccelerationPlugin.register_plugin(
    FastKernelsAccelerationPlugin,
    configuration_or_paths=[
        "training.fused_ops_and_kernels",
        "peft.quantization.fused_ops_and_kernels",
    ],
)