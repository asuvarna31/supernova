# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

import os
from dataclasses import dataclass, field
from datetime import datetime

import torch
import wandb
from datasets import load_dataset

from trl import (
    GRPOConfig,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from accuracy_reward_gr import accuracy_reward_gr, accuracy_reward_gr_with_length
from prefix_aware_grpo import PrefixAwareGRPOTrainer


@dataclass
class ScriptArguments:
    """adding script arg"""

    data_name: str = field(
        default="bbeh",
        metadata={"help": "name of dataset being trained on"},
    )

    target_size: int = field(
        default=500,
        metadata={"help": "number of training examples to use"},
    )


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

PROMPT = """{problem}

Think step by step, and when you are ready to provide the final answer, use the prefix "The answer is:" followed by the answer directly. For multi-choice questions, provide the letter, e.g. "The answer is: (a)".
"""

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 1. Get the model name (handling local paths or HF IDs)
    model_id = model_args.model_name_or_path.split("/")[-1]

    # 2. Format the learning rate (e.g., 1e-05)
    lr = training_args.learning_rate
    timestamp = datetime.now().strftime("%b%d-%H%M")

    # 3. Create a descriptive run name
    training_args.run_name = f"{model_id}-lr{lr}_{timestamp}_{script_args.data_name}"

    training_args.output_dir = f"{model_id}_{timestamp}_{script_args.data_name.split('/')[-1]}"
    os.makedirs(training_args.output_dir, exist_ok=True)

    os.environ["WANDB_PROJECT"] = "general_reasoning"  # Change this to your project name
    os.environ["WANDB_NAME"] = training_args.run_name

    def make_conversation(example):
        prompt = [
            {"role": "user", "content": PROMPT.format(problem=example["problem"])},
        ]
        return {"prompt": prompt}

    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        trust_remote_code=True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ###############
    # Dataset
    ###############
    dataset = load_dataset(script_args.data_name, split="train")
    if "problem" not in dataset.column_names:
        dataset = dataset.rename_column("question", "problem")
        dataset = dataset.rename_column("answer", "solution")
    dataset = dataset.map(make_conversation)
    dataset = dataset.rename_column("completions", "generations")

    target_size = script_args.target_size
    if len(dataset) < target_size:
        repeats = target_size // len(dataset)
        remainder = target_size % len(dataset)
        indices = list(range(len(dataset))) * repeats + list(range(remainder))
        dataset = dataset.select(indices)
        train_dataset = dataset.shuffle(seed=42)
    else:
        train_dataset = dataset.shuffle(seed=42).select(range(target_size))

    print(f"Train size: {len(train_dataset)}")

    ### Loading BBEH mini for eval ###
    bbeh = load_dataset("BBEH/bbeh", split="train")
    bbeh = bbeh.rename_column("input", "problem").rename_column("target", "solution")
    bbeh_mini = bbeh.filter(lambda x: x["mini"] == 1)
    bbeh_mini = bbeh_mini.map(make_conversation)
    bbeh_mini = bbeh_mini.map(lambda x: {**x, "eval_group": "bbeh"})

    eval_datasets = {"bbeh_mini": bbeh_mini}

    def combined_reward(completions, eval_group=None, **kwargs):
        if eval_group and eval_group[0] == "bbeh":
            return accuracy_reward_gr(completions, **kwargs)
        return accuracy_reward_gr_with_length(completions, **kwargs)

    #### Save best model based on bbeh_mini ####
    training_args.metric_for_best_model = "eval_bbeh_mini_reward"
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True

    ################
    # Training
    ################
    trainer = PrefixAwareGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[combined_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        peft_config=get_peft_config(model_args),
    )

    if wandb.run is not None:
        wandb.config.update({"train_dataset_size": len(train_dataset)})
    trainer.train()

    # Save the best model (loaded at end due to load_best_model_at_end=True)
    trainer.save_model(training_args.output_dir)