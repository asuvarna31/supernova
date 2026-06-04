"""Evaluation functions for ZebraLogic."""

import re
from typing import Dict, List, Tuple

KEY_ALIASES = {
    'bookgenre': 'bookgenre',
    'genre': 'bookgenre',
    'book': 'bookgenre',
    'phonemodel': 'phonemodel',
    'phone': 'phonemodel',
    'smoothie': 'smoothie',
    'education': 'education',
    'name': 'name',
    'house': 'house',
    'child': 'child',
    'children': 'child',
    'nationality': 'nationality',
    'food': 'food',
    'color': 'color',
    'colour': 'color',
    'animal': 'animal',
    'sport': 'sport',
    'sports': 'sport',
    'vacation': 'vacation',
    'car': 'car',
    'carmodel': 'car',
}


def normalize_key(key: str) -> str:
    key = re.sub(r'[^a-z0-9]', '', key.lower().strip())
    return KEY_ALIASES.get(key, key)


def normalize_value(value: str) -> str:
    value = value.lower().strip()
    value = value.rstrip('.')
    value = re.sub(r'\s+', ' ', value)  # collapse multiple spaces
    return value


def extract_answer(sample: str) -> str:
    answer_prefixes = [
        "The answer is:",
        "The final answer is:",
        "The final answer is ",
        "The answer is ",
    ]
    answer = sample
    for prefix in answer_prefixes:
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    return answer


def parse_house_string(house_str: str) -> Dict[str, str]:
    result = {}
    parts = house_str.strip().split(",")
    for part in parts:
        part = part.strip()
        if ":" in part:
            key, _, value = part.partition(":")
            result[normalize_key(key)] = normalize_value(value.strip())
    return result


def parse_solution(solution_str: str) -> List[Dict[str, str]]:
    """Handles both newline and ' | ' separators."""
    if '|' in solution_str:
        houses = solution_str.strip().split("|")
    else:
        houses = solution_str.strip().split("\n")
    return [parse_house_string(h) for h in houses if h.strip()]


def preprocess_sample(sample: str) -> str:
    answer = extract_answer(sample.strip())
    answer = answer.replace("**", "").strip()
    return answer.lower()


def preprocess_reference(reference: str) -> str:
    return reference.strip().lower()


def fuzzy_match(prediction: str, reference: str) -> bool:
    pred_houses = parse_solution(prediction)
    ref_houses = parse_solution(reference)

    if len(pred_houses) != len(ref_houses):
        return False

    for pred_house, ref_house in zip(pred_houses, ref_houses):
        # Only check keys that exist in the reference
        for key, ref_val in ref_house.items():
            if key == 'house':
                continue  # skip house number, check content only
            pred_val = pred_house.get(key)
            if pred_val != ref_val:
                return False

    return True


def compute_partial_score(prediction: str, reference: str) -> Tuple[float, int, int]:
    pred_houses = parse_solution(prediction)
    ref_houses = parse_solution(reference)

    total = len(ref_houses)
    if total == 0:
        return 0.0, 0, 0

    correct = 0
    for pred_house, ref_house in zip(pred_houses, ref_houses):
        house_correct = all(
            pred_house.get(k) == v
            for k, v in ref_house.items()
            if k != 'house'
        )
        if house_correct:
            correct += 1

    return correct / total, correct, total


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


def evaluate_partial(sample: str, reference: str) -> Tuple[float, int, int]:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return compute_partial_score(prediction, reference)