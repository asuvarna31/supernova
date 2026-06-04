import re
from typing import List, Dict, Optional

# --- Logic from evaluate.py from BBEH ---

def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response

def extract_answer(sample: str) -> str:
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)

def fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False

def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction

def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference

def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)

# --- Reward Function ---

def accuracy_reward_gr(completions: List[List[Dict[str, str]]], solution: List[str], **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution, strict=True):
        try:
            is_correct = evaluate_correctness(content, sol)
            rewards.append(1.0 if is_correct else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

from typing import List, Dict

def accuracy_reward_gr_with_length(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    alpha: float = 0.01,
    target_length: int = 100,
    **kwargs
) -> List[float]:
    """
    Reward function that combines correctness (1/0) with a length penalty.
    
    Args:
        completions: model outputs, each completion is a list of dicts with "content"
        solution: list of ground-truth answers
        alpha: scaling factor for length penalty
        target_length: desired number of words in completion
    Returns:
        List of floats: combined rewards
    """
    rewards = []
    for completion, sol in zip(completions, solution, strict=True):
        text = completion[0]["content"]
        
        # 1. Base reward: correctness
        try:
            is_correct = evaluate_correctness(text, sol)
            base_reward = 1.0 if is_correct else 0.0
        except Exception:
            base_reward = 0.0
        
        # 2. Length penalty reward
        length = len(text.split())  # count words
        length_reward = -alpha * abs(length - target_length)
        
        # 3. Combine rewards
        total_reward = base_reward + length_reward
        rewards.append(total_reward)
    
    return rewards

def accuracy_rewards_truncation_penalty(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs
) -> List[float]:
    rewards = []
    for completion, sol, comp_ids in zip(completions, solution, kwargs.get("completion_ids", []), strict=True):
        text = completion[0]["content"]
        
        # 1. Base reward: correctness
        try:
            is_correct = evaluate_correctness(text, sol)
            base_reward = 1.0 if is_correct else 0.0
        except Exception:
            base_reward = 0.0
        
        # 2. Truncation penalty
        is_truncated = len(comp_ids) >= 4096
        truncation_penalty = -0.5 if is_truncated else 0.0
        
        total_reward = base_reward + truncation_penalty
        rewards.append(total_reward)
    
    return rewards