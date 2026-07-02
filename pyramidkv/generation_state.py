"""Helpers for tracking per-generation KV-cache state.

The monkey-patched Llama/Mistral attention modules keep a per-layer
``kv_seq_len`` attribute to remember how many tokens have already been
processed. That attribute must be reset to ``0`` at the start of every
independent ``generate()`` call, otherwise stale sequence-length state leaks
from one example into the next and corrupts KV-cache compression (see issue
#46). ``is_empty_past_key_values`` centralizes the "is this a fresh cache?"
check used by the ``prepare_inputs_for_generation`` patches so both model
backends agree on the reset condition.
"""


def is_empty_past_key_values(past_key_values):
    """Return True when ``past_key_values`` holds no cached tokens yet.

    Handles the three shapes seen across transformers versions: ``None``, a
    ``Cache`` object exposing ``key_cache``, and the legacy tuple layout.
    """
    if past_key_values is None:
        return True
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        return len(key_cache) == 0
    if isinstance(past_key_values, tuple):
        return len(past_key_values) == 0
    return False
