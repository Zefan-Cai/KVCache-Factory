# KV Cache Algorithm Backlog

Evidence date: 2026-06-23.

This backlog tracks representative KV cache algorithms to keep or implement in KVCache-Factory and its nano-vllm / mini-sglang ports. GitHub star counts were checked with the GitHub API; citation counts are approximate and were checked with OpenAlex or Semantic Scholar when available.

## Selection Rules

- Prefer methods with high citations, high GitHub stars, or broad baseline usage.
- Prefer mechanism diversity over near-duplicate implementations.
- Implement first in KVCache-Factory, then port the same algorithm contract to the runtime forks.
- Record whether a method is already implemented, partially implemented, or missing.

## Current Coverage

| Category | Method | Representative signal | Status |
| --- | --- | --- | --- |
| Compression | StreamingLLM | `mit-han-lab/streaming-llm`: 7231 stars; ICLR 2024; attention sinks baseline | Implemented |
| Compression/retrieval | H2O | `FMInference/H2O`: 523 stars; NeurIPS 2023 heavy-hitter baseline | Implemented |
| Retrieval/compression | SnapKV | `FasterDecoding/SnapKV`: 321 stars; NeurIPS 2024; observation-window selection | Implemented |
| Retrieval | Quest | `mit-han-lab/quest`: 397 stars; ICML 2024; query-aware sparse KV retrieval | Implemented core page/token selector |
| Compression | NACL | ACL 2024; single-operation encoding-time eviction with proxy-token and random eviction | Implemented core proxy/random selector |
| Compression | Scissorhands | NeurIPS 2023; persistence-of-importance eviction baseline | Implemented core persistence selector |
| Cross-layer compression | MiniCache | NeurIPS 2024; Semantic Scholar: 85 citations; depth-dimension KV compression | Implemented core SLERP merge/restore contract |
| Compression/budgeting | PyramidKV | `Zefan-Cai/KVCache-Factory`: 1346 stars; project core method | Implemented |
| Adaptive compression | AdaKV | `FFY0/AdaKV`: 134 stars; NeurIPS 2025; head-adaptive budgets | Implemented |
| Adaptive retrieval | HeadKV | Semantic Scholar: 84 citations for arXiv:2410.19258; ICLR 2025 | Implemented |
| Retrieval/compression | L2Norm | EMNLP 2024; simple norm-based baseline | Implemented |
| Merge | LOOK-M pivot merge | EMNLP Findings 2024; `SUSTechBruce/LOOK-M`: 103 stars | Implemented (`--merge pivot`) |
| Merge | KVMerger-style weighted merge | OpenReview/arXiv 2024; adaptive token-level KV merging | Implemented nearest-neighbor weighted merge (`--merge weighted`) |
| Quantization | KIVI-style HQQ cache config | `jy-yuan/KIVI`: 411 stars; ICML 2024; key per-channel/value per-token axis defaults | Implemented config path (`--quant_method kivi`) |
| Quantization | KVQuant-style outlier path | `squeezeailab/kvquant`: 427 stars; NeurIPS 2024 | Partially implemented (`--quant_method kvquant`) |
| Sparse prefill | MInference | `microsoft/MInference`: 1220 stars; NeurIPS 2024 Spotlight | Partially integrated |

## Priority Candidates

| Priority | Category | Method | Evidence | Implementation target |
| --- | --- | --- | --- | --- |
| P0 | Retrieval runtime integration | Quest hot path | `mit-han-lab/quest`: 397 stars; ICML 2024; query-aware sparse KV retrieval | Wire the tested page/token selector into decode attention for KVCache-Factory, nano-vllm, and mini-sglang. |
| P1 | Merge | KVMerger parity | OpenReview/arXiv 2024; adaptive token-level KV merging | Replace the current nearest-neighbor weighted merge with the paper's full merge-set identification if needed for parity. |
| P1 | Quantization | KIVI kernel parity | `jy-yuan/KIVI`: 411 stars; ICML 2024; asymmetric 2-bit KV quantization | Replace or augment the HQQ config path with official-kernel-equivalent packing/dequantization if needed for performance parity. |
| P1 | Compression runtime integration | NACL hot path | ACL 2024; single-operation encoding-time eviction | Wire the tested proxy/random selector into prompt-time cache eviction for KVCache-Factory, nano-vllm, and mini-sglang. |
| P1 | Compression runtime integration | Scissorhands hot path | NeurIPS 2023; persistence-of-importance eviction baseline | Wire the tested historical-importance selector into decode-time cache eviction for KVCache-Factory, nano-vllm, and mini-sglang. |
| P1 | Cross-layer runtime integration | MiniCache hot path | NeurIPS 2024; depth-dimension KV compression | Wire the tested adjacent-layer merge/restore contract into paired-layer cache storage for KVCache-Factory, nano-vllm, and mini-sglang. |
| P1 | Quantization | GEAR | arXiv 2024; near-lossless KV compression recipe | Add low-rank/residual quantization prototype if it fits the current cache abstraction. |
| P2 | Systems compression | CacheGen | SIGCOMM 2024; 72 OpenAlex citations; `UChi-JCL/CacheGen`: 159 stars | Track for cache serialization/streaming rather than first-pass in-model attention. |
| P2 | Library baseline | NVIDIA kvpress | `NVIDIA/kvpress`: 1116 stars and active in 2026 | Use as an interoperability/reference baseline; do not blindly copy its API. |
| P2 | Quantization | TurboQuant | ICLR 2026; Google Research; strong new quantization idea but low citation age | Track after KIVI/KVQuant/GEAR unless a compact reference implementation becomes mature. |

## Runtime Porting Notes

- KVCache-Factory can use Hugging Face monkeypatches; nano-vllm and mini-sglang need runtime-native integration.
- Nano-vllm status: `nanovllm.kvcache_factory` now contains CPU-tested core selectors for StreamingLLM/H2O/SnapKV/Quest/NACL/Scissorhands, MiniCache cross-layer merge/restore utilities, nearest-token merge for LOOK-M/KVMerger-style modes, and KIVI/KVQuant config metadata. Next step: wire these contracts into block-table/cache updates without breaking prefix-cache invariants.
- Mini-sglang status: `minisgl.kvcache_factory` now contains the matching CPU-tested core selectors for StreamingLLM/H2O/SnapKV/Quest/NACL/Scissorhands, MiniCache cross-layer merge/restore utilities, nearest-token merge, and KIVI/KVQuant config metadata. Next step: wire these contracts into cache managers/attention backends while preserving radix-prefix sharing semantics.
- For nano-vllm, preserve block-table and cache allocation invariants before pruning or merging tokens.
- For mini-sglang, preserve prefix-sharing/radix-cache semantics before modifying cache contents.
- Every method needs at least synthetic shape/budget tests in all target repos before GPU benchmarking.
