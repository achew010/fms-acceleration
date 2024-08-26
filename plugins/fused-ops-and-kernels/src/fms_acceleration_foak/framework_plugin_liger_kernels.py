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
from typing import Dict, Tuple, Any
import torch
import torch.distributed as dist

from fms_acceleration import AccelerationPlugin
from transformers import TrainingArguments, AutoConfig, AutoModelForCausalLM
from liger_kernel.transformers.trainer_integration import _apply_liger_kernel
from liger_kernel.transformers import apply_liger_kernel_to_llama

class LigerKernelsAccelerationPlugin(AccelerationPlugin):

    # NOTE: may remove this when we have generic model rules
    restricted_model_archs = [
        "MixtralForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)
        self._check_config_equal(
            key="peft.enable_liger_kernels",
            value=True,
        )

    def model_loader(self, model_name: str, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        _apply_liger_kernel(config.model_type)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            **kwargs,
        )
        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_agumentation(self):
        return False

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        from .liger_utils import EfficiencyCallback
        return [
            EfficiencyCallback()
        ]


# register
AccelerationPlugin.register_plugin(
    LigerKernelsAccelerationPlugin,
    configuration_and_paths=[
        "peft.enable_liger_kernels"
    ],
)
