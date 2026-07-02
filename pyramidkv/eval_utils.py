"""Shared helpers for the evaluation entrypoints (run_longbench.py, run_ruler.py,
run_needle_in_haystack.py) so protocol fixes stay in one place."""

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false/1/0, got {value!r}")


def build_stop_token_ids(model, tokenizer):
    """Collect every stop token id (e.g. Llama-3 dual terminators, issue #46)."""
    stop_token_ids = []

    def _add(token_id):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)

    generation_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(generation_eos, (list, tuple)):
        for token_id in generation_eos:
            _add(token_id)
    else:
        _add(generation_eos)

    _add(tokenizer.eos_token_id)

    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            _add(token_id)

    return stop_token_ids
