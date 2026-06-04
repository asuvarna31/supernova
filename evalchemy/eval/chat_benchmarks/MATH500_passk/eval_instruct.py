import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from collections import defaultdict

import lm_eval.models
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\n
Think step by step, and when you are ready to provide the final answer,mark your solution with \\boxed\nAnswer:"""

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


class MATH500Benchmark(BaseBenchmark):
    """
    MATH500 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    Follows the evaluation logic of hendrycks_math answer extraction.
    Supports pass@k evaluation by generating multiple samples per question.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/MATH500/data/math500.jsonl",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        num_samples: int = 8,
        k_values: Optional[List[int]] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MATH500 benchmark.

        Args:
            data_file: File containing the MATH500 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            num_samples: Number of solutions to generate per question (default: 256)
            k_values: List of k values for pass@k. Default: [1, 2, 4, 8, 16, 32, 64, 128, 256]
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = 4096  # set higher to avoid truncation for reasoning models
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

        # Prepare instances for model
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
                {"role": "user", "content": PROMPT.format(problem=example["problem"])},
            ]

            templated_messages = self._prepare_messages(messages, model)

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
                                "temperature": 1,
                                "seed": sample_seed,
                            },
                        ),
                        instance_idx,
                    )
                )
                instance_idx += 1

        # Generate model responses
        self.logger.info(
            f"Generating {len(all_instances)} responses for MATH500 "
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
            question_answers = [self.extract_answer(o) for o in question_outputs]

            sample_correct = [
                is_equiv(str(example["answer"]), ans) for ans in question_answers
            ]

            example["model_outputs"] = question_outputs
            example["model_answers"] = question_answers
            example["sample_correct"] = sample_correct

        return {"examples": examples, "num_samples": self.num_samples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions with pass@k, per-subject and overall."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_samples = results["num_samples"]
        num_questions = len(examples)

        # --- Per-subject pass@k ---
        subject_pass_at_k = defaultdict(lambda: {k: [] for k in self.k_values})
        # --- Overall pass@k ---
        overall_pass_at_k = {k: [] for k in self.k_values}

        for example in examples:
            subject = example.get("subject", "unknown")
            num_correct = sum(example["sample_correct"])
            example["num_correct"] = num_correct
            example["num_samples"] = num_samples

            for k in self.k_values:
                p_at_k = estimate_pass_at_k(num_samples, num_correct, k)
                subject_pass_at_k[subject][k].append(p_at_k)
                overall_pass_at_k[k].append(p_at_k)

        # --- Aggregate per-subject metrics ---
        subject_results = {}
        for subject in sorted(subject_pass_at_k.keys()):
            subject_metrics = {}
            for k in self.k_values:
                subject_metrics[f"pass@{k}"] = float(np.mean(subject_pass_at_k[subject][k]))
            subject_results[subject] = subject_metrics

        # --- Aggregate overall metrics ---
        metrics = {
            "num_total": num_questions,
            "num_samples_per_question": num_samples,
        }

        for k in self.k_values:
            # Macro average: mean of per-subject means
            subject_means = [subject_results[s][f"pass@{k}"] for s in subject_results]
            metrics[f"pass@{k}_macro"] = float(np.mean(subject_means))

        metrics["accuracy"] = metrics["pass@1_macro"]
        metrics["subject_results"] = subject_results

        # --- Logging ---
        self.logger.info("=== MATH500 Results ===")
        for k in self.k_values:
            self.logger.info(
                f"  pass@{k} (macro): {metrics[f'pass@{k}_macro']:.4f}"
            )
        self.logger.info("Per-subject results:")
        for subject in sorted(subject_results.keys()):
            scores_str = ", ".join(
                f"pass@{k}: {subject_results[subject][f'pass@{k}']:.4f}" for k in self.k_values
            )
            self.logger.info(f"  {subject}: {scores_str}")

        results.update(metrics)
        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load MATH500 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, which is expected to be in the format of \\boxed{answer}.

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \\boxed.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""