# GQA KV-Cache Layout Audit (issue #49)

Evidence date: 2026-07-02.

This note answers the GQA cache-layout question in
[issue #49](https://github.com/Zefan-Cai/KVCache-Factory/issues/49) and records a
concrete, validation-gated migration plan. It is an audit + plan, not a landed
change: the correct refactor alters token-selection semantics and cannot be
verified without full benchmark reruns, so it must not be applied blindly.

## Current behavior

Every compression forward stores the KV cache at **query-head granularity**, not
KV-head granularity. The common pattern (e.g. `pyramidkv/llama_model.py:158`,
`:167`, `:168`) is:

```python
key_states   = repeat_kv(key_states, self.num_key_value_groups)   # KV heads -> query heads
value_states = repeat_kv(value_states, self.num_key_value_groups)
...
key_states_compress, value_states_compress = self.kv_cluster.update_kv(
    key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
```

So `repeat_kv` runs **before** compression, and the expanded, compressed tensor
is what gets written into the cache.

Scope of the pattern (grepped 2026-07-02):

| Location | `repeat_kv`-before-compress sites | `kv_cluster.update_kv` call sites |
| --- | --- | --- |
| `pyramidkv/llama_model.py` | 20 | 20 |
| `pyramidkv/mistral_model.py` | 20 | 20 |

Cluster classes with an `update_kv` that assumes query-head-granularity input
(`pyramidkv/pyramidkv_utils.py`): `PyramidKVCluster`, `SnapKVCluster`,
`L2NormCluster`, `CAMKVCluster`, `H2OKVCluster`, `StreamingLLMKVCluster`,
`AdaKVCluster`, `HeadKVCluster` — 8 classes.

## Memory implication

For a GQA model the stored cache is `num_key_value_groups` times larger than a
KV-head-granularity cache. For Meta-Llama-3-8B (32 query heads, 8 KV heads,
`num_key_value_groups = 4`), the compressed KV cache is **~4x** the size an
efficient GQA cache would use. This negates GQA's main KV-cache memory benefit:
the reported `max_capacity_prompt` budget is spent per query head rather than per
KV head. The reporter in #49 is correct on this point.

## Why this is not a mechanical refactor

Moving compression before `repeat_kv` (score/select at KV-head granularity, then
`repeat_kv` only for attention compute) **changes which tokens are kept**, so it
is a semantics change, not a behaviour-preserving refactor:

- Selection is currently **per query head**. `SnapKVCluster.update_kv`
  (`pyramidkv/pyramidkv_utils.py:378`) does `attn_cache.topk(...)` producing a
  distinct kept-token set for each of the 32 query heads; `H2OKVCluster`
  (`:593`) does the same on accumulated scores.
- At KV-head granularity there is only one cache slot per KV head, shared by its
  `num_key_value_groups` query heads. The 4 query heads in a group generally
  rank different tokens, so a single kept set per KV head requires **reducing**
  the 4 heads' scores (sum / mean / max) before top-k. That reduction is a new
  modeling choice that changes the retained tokens and therefore the model
  output and downstream accuracy.
- `AdaKVCluster` / `HeadKVCluster` add per-head *budgets* on top of per-head
  selection, so their semantics under head grouping need separate design.

In short: the "efficient" layout is not free — it trades a documented ~4x memory
win for a change in the selection algorithm that every method must re-justify.

## Migration plan (validation-gated)

Do not change all 40 call sites at once. Land incrementally, each step behind a
benchmark gate, because correctness can only be shown on GPU (LongBench / RULER),
not by unit tests alone.

1. **Pin current behavior.** DONE (2026-07-02, 8xA100-40G): 6-task LongBench
   subset (multifieldqa_en/hotpotqa/triviaqa/gov_report/samsum/lcc),
   Meta-Llama-3-8B-Instruct @128, fp16, flash-attn2, greedy. Pre-fix code
   (`76b07dc`): FullKV 41.15/45.15/90.56/28.68/42.67/59.38, SnapKV
   31.30/40.41/89.78/19.88/38.99/59.00, PyramidKV
   32.90/38.29/89.23/20.29/38.96/57.74. On fixed main (`7855979`, Llama-3
   stop-token + chat-template fixes from issue #46) the query_head numbers
   are: FullKV 47.78/47.32/90.56/30.44/42.67/59.38, SnapKV
   45.08/45.94/89.78/21.01/38.99/59.00, PyramidKV
   44.57/46.03/89.23/21.02/38.96/57.74.
2. **Prototype behind a flag.** DONE (2026-07-01): `--kv_cache_granularity
   {query_head,kv_head}` (default `query_head`) and `--gqa_score_agg
   {mean,max,sum}` (default `mean`) landed for SnapKV, PyramidKV, H2O,
   StreamingLLM, CAM and L2Norm on both Llama and Mistral. Mode is detected
   inside `update_kv` from query-vs-key head counts; scores are group-reduced
   before top-k; `repeat_kv` moved to attention-compute time (eager/sdpa) or
   dropped entirely (flash-attn handles GQA natively). Default-mode outputs are
   bit-identical to the pre-change code (tests/test_query_head_bitident.py);
   kv_head invariants and tiny-model generation are covered by
   tests/test_gqa_kv_head.py and tests/test_gqa_model_integration.py.
3. **Prove parity/tradeoff.** DONE (2026-07-02, same setup, `--gqa_score_agg
   mean`): kv_head vs query_head on the 6-task subset — SnapKV
   46.10/46.19/90.37/20.11/39.51/59.30 vs 45.08/45.94/89.78/21.01/38.99/59.00
   (avg +0.3), PyramidKV 44.63/46.36/89.32/20.36/38.20/57.51 vs
   44.57/46.03/89.23/21.02/38.96/57.74 (wash), StreamingLLM bit-identical
   (position-based selection is granularity-invariant). Peak memory
   (`scripts/benchmark_latency_memory.py`, 9k-token prompt, SnapKV,
   flash-attn2, A100-40G): budget 2048 peak allocated 22.49 -> 21.74 GiB
   (-748 MiB, the expected 4x shrink of the compressed cache itself); budget
   128 delta ~48 MiB (prefill activations dominate). Decode slightly faster
   in kv_head mode (25.6 vs 25.4 tok/s @2048). Accuracy parity accepted; the
   flag still defaults to `query_head` so published numbers stay
   reproducible.
4. **Generalize.** `AdaKV`/`HeadKV`/`ThinK` still need per-head-budget redesign
   and remain query-head-granular; `run_longbench.py` rejects
   `--kv_cache_granularity kv_head` for them at startup.

The query-head-granularity layout stays the default. This matches the
maintainer's note on #49 that a correct refactor needs a wider audit rather
than a risky local change.
