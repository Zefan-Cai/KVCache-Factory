import argparse
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def read_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
    else:
        prompt = args.prompt
    return prompt * args.prompt_repeats


def apply_monkeypatch(model_path, method):
    method = method.lower()
    if method == "fullkv":
        return

    from pyramidkv.monkeypatch import replace_llama, replace_mistral

    model_path = model_path.lower()
    if "mistral" in model_path:
        replace_mistral(method)
    else:
        replace_llama(method)


def configure_kv_method(model, args):
    if args.method.lower() == "fullkv":
        return

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return

    for layer in model.model.layers:
        if not hasattr(layer, "self_attn"):
            continue
        config = layer.self_attn.config
        config.max_capacity_prompt = args.max_capacity_prompt
        config.kernel_size = args.kernel_size
        config.pooling = args.pooling
        config.merge = args.merge
        if args.method.lower() == "streamingllm":
            config.window_size = args.max_capacity_prompt - args.start_size
        else:
            config.window_size = args.window_size
        if hasattr(args, "recent_size"):
            config.recent_size = args.recent_size
        if hasattr(args, "pruning_ratio"):
            config.ratio = args.pruning_ratio


def cuda_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def run_once(model, tokenizer, prompt, args):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    synchronize()
    start = time.perf_counter()
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": 1,
        "use_cache": True,
    }
    if args.do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    output = model.generate(**inputs, **generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = output.shape[1]
    generated_tokens = total_tokens - input_tokens
    stats = {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else None,
    }
    stats.update(cuda_stats())
    return stats


def main(args):
    set_seed(args.seed)
    prompt = read_prompt(args)
    apply_monkeypatch(args.model_path, args.method)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    configure_kv_method(model, args)

    for _ in range(args.warmup):
        run_once(model, tokenizer, prompt, args)

    runs = [run_once(model, tokenizer, prompt, args) for _ in range(args.repeat)]
    result = {
        "model_path": args.model_path,
        "method": args.method,
        "attn_implementation": args.attn_implementation,
        "max_capacity_prompt": args.max_capacity_prompt,
        "window_size": args.window_size,
        "max_new_tokens": args.max_new_tokens,
        "runs": runs,
    }
    valid_tps = [run["tokens_per_sec"] for run in runs if run["tokens_per_sec"] is not None]
    if valid_tps:
        result["mean_tokens_per_sec"] = float(np.mean(valid_tps))
    if torch.cuda.is_available():
        result["max_peak_allocated_gib"] = max(run["peak_allocated_gib"] for run in runs)
        result["max_peak_reserved_gib"] = max(run["peak_reserved_gib"] for run in runs)

    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--method", default="fullkv")
    parser.add_argument("--attn_implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)
    parser.add_argument("--prompt", default="Summarize the key idea of KV cache compression in one paragraph.\n")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--prompt_repeats", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", default="maxpool")
    parser.add_argument("--merge", default=None)
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--pruning_ratio", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
