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

# Local
from fms_acceleration.model_patcher import ModelPatcher
from functools import partial

PATCHES = [".models.llama", ".models.mistral", ".models.mixtral"]
PLUGIN_PREFIX = "fms_acceleration_foak"

# TODO: remove the need for the prefix
def register_foak_model_patch_rules(base_type):
    ignored_rule_names = ModelPatcher.rules
    ModelPatcher.load_patches(
        [f"{PLUGIN_PREFIX}{postfix}" for postfix in PATCHES],
    )
    for rule_name, rule in ModelPatcher.rules.items():
        if rule_name not in ignored_rule_names and rule.forward_builder is not None:
            rule.forward_builder = partial(rule.forward_builder, base_type=base_type)
