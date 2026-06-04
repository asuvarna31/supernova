import os
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import hmean
import random

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from collections import defaultdict
from eval.task import BaseBenchmark
from eval.chat_benchmarks.BBH_mini.evaluate import preprocess_sample, preprocess_reference, fuzzy_match

import lm_eval.models
from lm_eval.models.vllm_causallms import VLLM

PROMPT = """{problem}

Think step by step, and when you are ready to provide the final answer, use the prefix "The answer is:" followed by the answer directly,  with no formatting and no markup. For instance: "The answer is: 42", or  "The answer is: yes", or "The answer is: (a)" For multi-choice questions, provide the letter, e.g. "The answer is: (a)
"""

# Default k values for pass@k
DEFAULT_K_VALUES = [1, 2, 4, 8]


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    """Estimate pass@k using the unbiased estimator from the Codex paper.

    pass@k = 1 - C(n - c, k) / C(n, k)

    Uses the numerically stable version:
    1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)

    Args:
        num_samples: Total number of generated samples (n)
        num_correct: Number of correct samples (c)
        k: Number of samples to consider

    Returns:
        Estimated pass@k probability
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


class BBHBenchmark(BaseBenchmark):
    """
    BBH Benchmark for evaluating the math reasoning of LLMs.
    Link: https://github.com/google-deepmind/BBH/tree/main

    Supports pass@k evaluation by generating multiple samples per question.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/BBH_mini/data/bbh_mini.json",
        seed: List[int] = [0, 1234, 1234, 1234],
        num_samples: int = 8,
        k_values: Optional[List[int]] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize BBH benchmark.

        Args:
            data_file: File containing the BBH dataset (id, problem, reference_solution, expected_answer, source)
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            num_samples: Number of solutions to generate per question (default: 8)
            k_values: List of k values for pass@k. Default: [1, 2, 4, 8]
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.seed = seed
        self.max_new_tokens = 4096
        self.num_samples = num_samples
        self.k_values = k_values or DEFAULT_K_VALUES
        # Filter k_values to only include those <= num_samples
        self.k_values = [k for k in self.k_values if k <= self.num_samples]

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate multiple solution completions per question using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and metadata,
            or None for non-primary ranks
        """
        examples = self.load_questions()

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        # Build instances: num_samples per question, each with a unique seed
        all_instances = []
        instance_idx = 0
        for idx, example in enumerate(examples):
            messages = [
                {"role": "user", "content": PROMPT.format(problem=example["input"])},
            ]
            templated_messages = model.apply_chat_template(messages)

            for sample_idx in range(self.num_samples):
                # Unique seed per sample to prevent deduplication
                sample_seed = [self.seed[0] + sample_idx] + self.seed[1:]
                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            templated_messages,
                            {
                                "do_sample": True,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 0.7,
                                "seed": sample_seed,
                            },
                        ),
                        instance_idx,
                    )
                )
                instance_idx += 1

        # Generate model responses
        self.logger.info(
            f"Generating {len(all_instances)} responses for BBH "
            f"({len(examples)} questions x {self.num_samples} samples)..."
        )
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Group outputs by question (num_samples consecutive outputs per question)
        for q_idx, example in enumerate(examples):
            start = q_idx * self.num_samples
            end = start + self.num_samples
            question_outputs = outputs[start:end]
            question_answers = [preprocess_sample(o) for o in question_outputs]

            processed_target = preprocess_reference(example["target"])
            sample_correct = [
                fuzzy_match(ans, processed_target) for ans in question_answers
            ]

            example["model_outputs"] = question_outputs
            example["model_answers"] = question_answers
            example["sample_correct"] = sample_correct

        return {"examples": examples, "num_samples": self.num_samples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions with pass@k, per-task and overall."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_samples = results["num_samples"]
        num_questions = len(examples)

        # --- Per-task pass@k ---
        task_pass_at_k = defaultdict(lambda: {k: [] for k in self.k_values})
        # --- Overall pass@k ---
        overall_pass_at_k = {k: [] for k in self.k_values}

        for example in examples:
            task = example["task"]
            num_correct = sum(example["sample_correct"])
            example["num_correct"] = num_correct
            example["num_samples"] = num_samples

            for k in self.k_values:
                p_at_k = estimate_pass_at_k(num_samples, num_correct, k)
                task_pass_at_k[task][k].append(p_at_k)
                overall_pass_at_k[k].append(p_at_k)

        # --- Aggregate per-task metrics ---
        task_results = {}
        for task in sorted(task_pass_at_k.keys()):
            task_metrics = {}
            for k in self.k_values:
                task_metrics[f"pass@{k}"] = float(np.mean(task_pass_at_k[task][k]))
            task_results[task] = task_metrics

        # --- Aggregate overall metrics (micro average across tasks) ---
        metrics = {
            "num_total": num_questions,
            "num_samples_per_question": num_samples,
        }

        for k in self.k_values:
            # Micro average: mean over all questions
            # metrics[f"pass@{k}"] = float(np.mean(overall_pass_at_k[k]))

            # Macro average: mean of per-task means
            task_means = [task_results[t][f"pass@{k}"] for t in task_results]
            metrics[f"pass@{k}_macro"] = float(np.mean(task_means))

            # # Harmonic mean of per-task pass@k (add 1 to avoid zero, matching original convention)
            # task_scores_for_hmean = [task_results[t][f"pass@{k}"] * 100 + 1 for t in task_results]
            # metrics[f"pass@{k}_harmonic"] = float(hmean(task_scores_for_hmean))

        metrics["accuracy"] = metrics["pass@1_macro"]
        metrics["task_results"] = task_results

        # --- Logging ---
        self.logger.info("=== asuvarna31 Results ===")
        for k in self.k_values:
            self.logger.info(
                # f"  pass@{k}: {metrics[f'pass@{k}']:.4f} "
                f"(macro: {metrics[f'pass@{k}_macro']:.4f}, "
                # f"harmonic: {metrics[f'pass@{k}_harmonic']:.4f})"
            )
        self.logger.info("Per-task results:")
        for task in sorted(task_results.keys()):
            scores_str = ", ".join(
                f"pass@{k}: {task_results[task][f'pass@{k}']:.4f}" for k in self.k_values
            )
            self.logger.info(f"  {task}: {scores_str}")

        results.update(metrics)
        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load BBH questions from the data file."""
        questions = []
        with open(self.data_file, "r") as f:
            questions = json.load(f)
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        random.shuffle(questions)
        return questions