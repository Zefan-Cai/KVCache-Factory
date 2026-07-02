"""Tiny CPU end-to-end generation checks for the flag-gated GQA kv-head path.

Builds randomly-initialised 2-layer Llama / Mistral models (eager attention),
monkeypatches the compression forwards exactly as run_longbench.py does, and
asserts the per-layer cache head count and length for both cache
granularities. The monkeypatched forwards mutate the transformers attention
classes globally, so every test re-applies the patch it needs before building
its model.
"""

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

from pyramidkv.monkeypatch import replace_llama, replace_mistral

WINDOW = 8
CAPACITY = 32
KERNEL = 7
POOLING = "maxpool"
PROMPT_LEN = 100
NEW_TOKENS = 5
NUM_LAYERS = 2
NUM_HEADS = 8


def _llama_config(num_kv_heads, attn_impl="eager"):
    return LlamaConfig(
        hidden_size=64,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=512,
        attn_implementation=attn_impl,
    )


def _mistral_config(num_kv_heads, attn_impl="eager"):
    return MistralConfig(
        hidden_size=64,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=512,
        attn_implementation=attn_impl,
    )


def _configure_layers(model, granularity, gqa_score_agg="mean"):
    """Thread the per-layer knobs onto self_attn.config the way run_longbench.py does."""
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i].self_attn.config.window_size = WINDOW
        layers[i].self_attn.config.max_capacity_prompt = CAPACITY
        layers[i].self_attn.config.kernel_size = KERNEL
        layers[i].self_attn.config.pooling = POOLING
        layers[i].self_attn.config.merge = None
        layers[i].self_attn.config.kv_cache_granularity = granularity
        layers[i].self_attn.config.gqa_score_agg = gqa_score_agg


def _generate(model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=0,
        )
    self_len = out.sequences.shape[-1]
    assert self_len == PROMPT_LEN + NEW_TOKENS, self_len
    return out


def _cache_tensors(past_key_values):
    """Return the per-layer key cache tensors from a Cache or legacy tuple."""
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.key_cache)
    return [layer[0] for layer in past_key_values]


def _pyramidkv_scheduled_capacities(q_len):
    """Replicate PyramidKVCluster's per-layer kept-token schedule (long branch)."""
    beta = 20
    min_num = (CAPACITY - WINDOW) // beta
    max_num = (CAPACITY - WINDOW) * 2 - min_num
    if max_num >= q_len - WINDOW:
        max_num = q_len - WINDOW
        min_num = (CAPACITY - WINDOW) * 2 - max_num
    steps = (max_num - min_num) // (NUM_LAYERS - 1)
    caps = []
    for layer_idx in range(NUM_LAYERS):
        if q_len < (CAPACITY - WINDOW) * 2:
            caps.append(CAPACITY - WINDOW)
        else:
            caps.append(max_num - layer_idx * steps)
    return caps


class LlamaGQAIntegrationTest(unittest.TestCase):
    def _run(self, method, granularity, num_kv_heads=2, attn_impl="eager"):
        replace_llama(method)
        torch.manual_seed(0)
        model = LlamaForCausalLM(_llama_config(num_kv_heads, attn_impl)).eval()
        _configure_layers(model, granularity)
        out = _generate(model)
        return _cache_tensors(out.past_key_values)

    def _expected_heads(self, granularity, num_kv_heads):
        return num_kv_heads if granularity == "kv_head" else NUM_HEADS

    def _check_flat_capacity(self, caches, granularity, num_kv_heads=2):
        for layer_idx, key_cache in enumerate(caches):
            self.assertEqual(key_cache.shape[1], self._expected_heads(granularity, num_kv_heads),
                             f"layer {layer_idx} head count in {granularity} mode")
            self.assertLessEqual(key_cache.shape[2], CAPACITY + 1 + NEW_TOKENS,
                                 f"layer {layer_idx} cache length in {granularity} mode")
            self.assertEqual(key_cache.shape[0], 1)

    def test_snapkv_kv_head(self):
        caches = self._run("snapkv", "kv_head")
        self._check_flat_capacity(caches, "kv_head")

    def test_snapkv_query_head(self):
        caches = self._run("snapkv", "query_head")
        self._check_flat_capacity(caches, "query_head")

    def test_snapkv_kv_head_mha_config(self):
        # MHA model (kv_heads == num_heads): kv_head mode degenerates to the
        # legacy path; cache keeps all 8 heads.
        caches = self._run("snapkv", "kv_head", num_kv_heads=NUM_HEADS)
        self._check_flat_capacity(caches, "kv_head", num_kv_heads=NUM_HEADS)

    def test_snapkv_query_head_mha_config(self):
        caches = self._run("snapkv", "query_head", num_kv_heads=NUM_HEADS)
        self._check_flat_capacity(caches, "query_head", num_kv_heads=NUM_HEADS)

    def test_streamingllm_both_modes(self):
        for granularity in ("kv_head", "query_head"):
            caches = self._run("streamingllm", granularity)
            self._check_flat_capacity(caches, granularity)

    def test_snapkv_sdpa_both_modes(self):
        # The sdpa forward has its own post-cache repeat path; cover it on CPU.
        for granularity in ("kv_head", "query_head"):
            caches = self._run("snapkv", granularity, attn_impl="sdpa")
            self._check_flat_capacity(caches, granularity)

    def test_pyramidkv_both_modes(self):
        # PyramidKV's per-layer schedule may keep more than CAPACITY tokens in
        # early layers (by design, in BOTH modes); bound each layer by its own
        # scheduled capacity + window instead of the flat cap.
        scheduled = _pyramidkv_scheduled_capacities(PROMPT_LEN)
        for granularity in ("kv_head", "query_head"):
            caches = self._run("pyramidkv", granularity)
            for layer_idx, key_cache in enumerate(caches):
                self.assertEqual(key_cache.shape[1], self._expected_heads(granularity, 2),
                                 f"layer {layer_idx} head count in {granularity} mode")
                self.assertLessEqual(key_cache.shape[2],
                                     scheduled[layer_idx] + WINDOW + 1 + NEW_TOKENS,
                                     f"layer {layer_idx} cache length in {granularity} mode")

    def test_pyramidkv_kept_length_matches_across_modes(self):
        replace_llama("pyramidkv")
        lengths = {}
        for granularity in ("kv_head", "query_head"):
            caches = self._run("pyramidkv", granularity)
            lengths[granularity] = [key_cache.shape[2] for key_cache in caches]
        self.assertEqual(lengths["kv_head"], lengths["query_head"])


class MistralGQAIntegrationTest(unittest.TestCase):
    def _run(self, granularity, num_kv_heads=2, attn_impl="eager"):
        replace_mistral("snapkv")
        torch.manual_seed(0)
        model = MistralForCausalLM(_mistral_config(num_kv_heads, attn_impl)).eval()
        _configure_layers(model, granularity)
        out = _generate(model)
        return _cache_tensors(out.past_key_values)

    def test_snapkv_kv_head(self):
        caches = self._run("kv_head")
        for layer_idx, key_cache in enumerate(caches):
            self.assertEqual(key_cache.shape[1], 2, f"layer {layer_idx}")
            self.assertLessEqual(key_cache.shape[2], CAPACITY + 1 + NEW_TOKENS)

    def test_snapkv_query_head(self):
        caches = self._run("query_head")
        for layer_idx, key_cache in enumerate(caches):
            self.assertEqual(key_cache.shape[1], NUM_HEADS, f"layer {layer_idx}")
            self.assertLessEqual(key_cache.shape[2], CAPACITY + 1 + NEW_TOKENS)

    def test_snapkv_sdpa_both_modes(self):
        # The sdpa forward has its own post-cache repeat path; cover it on CPU.
        for granularity, heads in (("kv_head", 2), ("query_head", NUM_HEADS)):
            caches = self._run(granularity, attn_impl="sdpa")
            for layer_idx, key_cache in enumerate(caches):
                self.assertEqual(key_cache.shape[1], heads, f"layer {layer_idx} ({granularity})")
                self.assertLessEqual(key_cache.shape[2], CAPACITY + 1 + NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()
