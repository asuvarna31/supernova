"""Evaluation functions for BigBench Hard."""

import re

# Task-specific answer regexes covering all gold answer formats
BBH_ANSWER_REGEX = {
    "boolean_expressions": "[tT]rue|[fF]alse",
    "causal_judgement": "[yY]es|[nN]o",
    "date_understanding": "MC",
    "disambiguation_qa": "MC",
    "dyck_languages": "[\\]\\)\\}\\> ]+",
    "formal_fallacies": "[iI]nvalid|[vV]alid",
    "geometric_shapes": "MC",
    "hyperbaton": "MC",
    "logical_deduction_five_objects": "MC",
    "logical_deduction_seven_objects": "MC",
    "logical_deduction_three_objects": "MC",
    "movie_recommendation": "MC",
    "multistep_arithmetic_two": "-?\\d+",
    "navigate": "[nN]o|[yY]es",
    "object_counting": "\\d+",
    "penguins_in_a_table": "MC",
    "reasoning_about_colored_objects": "MC",
    "ruin_names": "MC",
    "salient_translation_error_detection": "MC",
    "snarks": "MC",
    "sports_understanding": "[yY]es|[nN]o",
    "temporal_sequences": "MC",
    "tracking_shuffled_objects_five_objects": "MC",
    "tracking_shuffled_objects_seven_objects": "MC",
    "tracking_shuffled_objects_three_objects": "MC",
    "web_of_lies": "[yY]es|[nN]o",
    "word_sorting": "[a-z ]+",
}


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


def extract_answer_with_task_regex(continuation: str, task: str = None) -> str:
    """Extract answer using task-specific regex patterns.

    Uses a cascade of increasingly permissive regexes, taking the LAST match
    of the first successful regex to handle CoT outputs where earlier text
    may coincidentally match.
    """
    answer_regex = "MC"
    if task and task in BBH_ANSWER_REGEX:
        answer_regex = BBH_ANSWER_REGEX[task]
    if answer_regex == "MC":
        answer_regex = "\\([A-Z]\\)"

    # Ordered from most specific to least specific
    regexes = [
        "(?i)So the answer is ($ANS$)\\.?",
        "(?i)the answer is:?\\s*($ANS$)",
        "(?i)the final answer is:?\\s*($ANS$)",
        "(?i)answer:.*?($ANS$)",
        "(?i)answer\\b.*?($ANS$)",
    ]

    # Last resort for MC tasks: standalone capital letter
    if answer_regex == "\\([A-Z]\\)":
        regexes.append("\\b([A-Z])\\b")

    # Final fallback: case-insensitive match anywhere
    regexes.append("(?i)($ANS$)")

    extracted_answer = ""
    for regex in regexes:
        regex = regex.replace("$ANS$", answer_regex)
        found = re.findall(regex, continuation)
        if found:
            extracted_answer = found[-1]  # Take the LAST match
            break

    # Strip common formatting delimiters
    special_delimiters_to_strip = [
        ("$", "$"),
        ("\\(", "\\)"),
        ("(", ")"),
        ("**", "**"),
        ("***", "***"),
        ("\\[", "\\]"),
        ("'", "'"),
        ("`", "`"),
        ('"', '"'),
    ]
    for left, right in special_delimiters_to_strip:
        if re.match(answer_regex, left):
            continue  # Don't strip valid answer chars (e.g. dyck_languages brackets)
        left_regex = re.escape(left)
        right_regex = re.escape(right)
        extracted_answer = re.sub(
            f"^{left_regex}(.*){right_regex}$", "\\1", extracted_answer
        ).strip()

    # Normalize single-letter MC answers to parenthesized form: A -> (A)
    if answer_regex == "\\([A-Z]\\)" and len(extracted_answer) == 1 and extracted_answer.isalpha():
        extracted_answer = f"({extracted_answer})"

    return extracted_answer


def preprocess_sample(sample: str, task: str = None) -> str:
    """Extracts and preprocesses the model's answer from its full output.

    Strategy 1: Task-aware regex extraction (if task is provided).
    Strategy 2: Generic prefix-based extraction searching from end of output.
    """
    text = sample.strip()

    # --- Strategy 1: Task-aware regex extraction (preferred) ---
    if task:
        answer = extract_answer_with_task_regex(text, task)
        if answer:
            answer = answer.lower().replace(", ", ",")
            return strip_latex(answer)

    # --- Strategy 2: Generic prefix-based extraction (case-insensitive, from end) ---
    text_lower = text.lower()
    answer_prefixes = [
        "the answer is:",
        "the final answer is:",
        "so the answer is:",
        "the answer is",
        "the final answer is",
        "so the answer is",
    ]

    best_idx = -1
    best_prefix_len = 0
    for prefix in answer_prefixes:
        idx = text_lower.rfind(prefix)
        if idx != -1 and (idx > best_idx or (idx == best_idx and len(prefix) > best_prefix_len)):
            best_idx = idx
            best_prefix_len = len(prefix)

    if best_idx != -1:
        prediction = text[best_idx + best_prefix_len:]
    else:
        # Fallback: return full text lowered (will likely fail to match)
        prediction = text

    prediction = prediction.strip().lower()
    prediction = prediction.replace("**", "")
    prediction = prediction.replace(", ", ",")
    prediction = prediction.split("\n")[0].strip()
    prediction = prediction.rstrip(".")
    prediction = strip_latex(prediction)
    return prediction


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Hard."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        if prediction[1] == reference:
            return True
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        if reference[1] == prediction:
            return True

    # Case-insensitive comparison
    if prediction.lower() == reference.lower():
        return True

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    # Strip all punctuation and compare
    pred_clean = re.sub(r"[^\w\s]", "", prediction).strip()
    ref_clean = re.sub(r"[^\w\s]", "", reference).strip()
    if pred_clean and pred_clean == ref_clean:
        return True

    return False


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str, task: str = None) -> bool:
    prediction = preprocess_sample(sample, task=task)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)