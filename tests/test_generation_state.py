import unittest

from pyramidkv.generation_state import is_empty_past_key_values


class _FakeCache:
    """Minimal stand-in for a transformers Cache exposing ``key_cache``."""

    def __init__(self, num_layers=0):
        self.key_cache = [object() for _ in range(num_layers)]


class _FakeLayer:
    def __init__(self):
        self.self_attn = type("Attn", (), {"kv_seq_len": 0})()


def _reset_if_fresh(layers, past_key_values):
    """Mirror the prepare_inputs_for_generation reset guard used by the patches.

    Returns True when a reset happened so tests can assert the branch.
    """
    if is_empty_past_key_values(past_key_values):
        for layer in layers:
            layer.self_attn.kv_seq_len = 0
        return True
    return False


class IsEmptyPastKeyValuesTest(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertTrue(is_empty_past_key_values(None))

    def test_empty_and_populated_cache_object(self):
        self.assertTrue(is_empty_past_key_values(_FakeCache(num_layers=0)))
        self.assertFalse(is_empty_past_key_values(_FakeCache(num_layers=4)))

    def test_empty_and_populated_legacy_tuple(self):
        self.assertTrue(is_empty_past_key_values(tuple()))
        self.assertFalse(is_empty_past_key_values(((1, 2),)))

    def test_object_without_key_cache_is_not_empty(self):
        # A non-empty cache-like object that does not expose ``key_cache`` and
        # is not a tuple should not be treated as fresh.
        self.assertFalse(is_empty_past_key_values(object()))


class KvSeqLenResetTest(unittest.TestCase):
    def test_fresh_call_resets_stale_kv_seq_len(self):
        # Simulate stale per-layer state left over from a previous generate().
        layers = [_FakeLayer() for _ in range(3)]
        for layer in layers:
            layer.self_attn.kv_seq_len = 512

        did_reset = _reset_if_fresh(layers, None)

        self.assertTrue(did_reset)
        self.assertTrue(all(layer.self_attn.kv_seq_len == 0 for layer in layers))

    def test_ongoing_decode_step_preserves_kv_seq_len(self):
        # Mid-generation (non-empty cache) must NOT reset the counter.
        layers = [_FakeLayer() for _ in range(3)]
        for layer in layers:
            layer.self_attn.kv_seq_len = 512

        did_reset = _reset_if_fresh(layers, _FakeCache(num_layers=3))

        self.assertFalse(did_reset)
        self.assertTrue(all(layer.self_attn.kv_seq_len == 512 for layer in layers))


if __name__ == "__main__":
    unittest.main()
