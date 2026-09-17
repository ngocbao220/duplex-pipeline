# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

import re
import json
import ast
from typing import List
from g2pk import G2p

# English pattern for Korean transliteration
ENG_PATTERN = re.compile(r"[A-Za-z]+")

# Cost calculation constants (for LLM usage tracking)
COST_PER_MILLION_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
}

COST_PER_MILLION_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4-turbo": 30.00,
    "gpt-3.5-turbo": 1.50,
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of API usage based on model name and token counts.

    Args:
        model_name: Name of the LLM model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    input_cost_per_million = COST_PER_MILLION_INPUT.get(model_name, 0.0)
    output_cost_per_million = COST_PER_MILLION_OUTPUT.get(model_name, 0.0)

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million

    total_cost = input_cost + output_cost
    return total_cost


def speaker_tagged_text(data):
    """
    Generate speaker-tagged text from segment data.

    Args:
        data: List of segments with 'speaker' and 'text' fields

    Returns:
        String with speaker tags and text
    """
    result = []
    for item in data:
        speaker = item.get("speaker", "Unknown")
        text = item.get("text", "")
        result.append(f"[{speaker}]: {text}")
    return "\n".join(result)


def parse_speaker_summary(llm_output: str) -> list | None:
    """
    Extract and parse a JSON array from the LLM output string.
    Handles 'json' prefix, code blocks (```), leading/trailing whitespace, etc.
    """
    if not llm_output:
        return None

    try:
        # Remove code blocks such as ```json ... ``` or ``` ... ```
        # Use a regular expression to find content between square brackets '[' and ']'
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Convert the JSON string to a Python object (list of dicts)
            return json.loads(json_str)
        else:
            print("Parsing Error: Could not find a valid JSON array format ([]).")
            return None

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unknown parsing error: {e}")
        return None


def process_llm_diarization_output(llm_output: str) -> list[dict]:
    """
    Process LLM output for diarization results.

    Args:
        llm_output: Raw LLM output string

    Returns:
        List of dictionaries containing diarization data
    """
    # 1. Find ```json ... ``` code block in LLM output
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # If no ```json block is found, attempt to parse the entire string
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # 2. Parse the JSON string into a Python object
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # In case the LLM output is in Python list format ('[{"text":...}]')
        try:
            # ast.literal_eval is a more secure version of eval.
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"Error: Failed to parse as both JSON and Python literal. {e!r}")
            return []

    return llm_data


def ko_transliterate_english(text: str) -> str:
    """
    Find English segments in the input string and convert them to Korean pronunciation.

    Args:
        text: Input text with English words

    Returns:
        Text with English transliterated to Korean pronunciation
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2p(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: List[dict]) -> None:
    """
    Process JSON list to transliterate English to Korean.

    Args:
        input_list: List of dictionaries with 'text' field
    """
    for entry in input_list:
        text = entry.get("text", "")
        # Convert if text contains English
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)
