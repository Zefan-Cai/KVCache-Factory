import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyramidkv.eval_utils import str2bool, build_stop_token_ids
from typing import List
from pyramidkv.quantization import build_quantized_cache_config, patch_quantized_cache


context_length_list = [4096]
# context_length_list = [4096, 8192, 16384]

datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7500,
    "llama-3": 7500,
    "mistral": 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

# def build_prompt(prompt, dataset):

#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []
    
    input_max_len = 0
    model_path = args.model_path.lower()
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:

            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: 
                input_max_len = length

            prompt = example["input"] #TODO tokenizer.apply_chat_template ?
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for example in test_data:
        prompt_list.append(example["prompt"])
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example["index"])

    print("Finish loading model and tokenizer")
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", str(args.context_length), args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", str(args.context_length), args.dataset, f"{args.method}.json"), "w")
    
    for i in tqdm(range(0, len(prompt_list), args.eval_batch_size)):
        
        batch_prompts = prompt_list[i:i+args.eval_batch_size]
        batch_inputs = input_list[i:i+args.eval_batch_size]
        batch_answers = outputs_list[i:i+args.eval_batch_size]
        batch_lengths = length_list[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        
        if args.method.lower() != "fullkv":
            if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm"]:
                window_sizes = 8
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            kernel_sizes = 7
            pooling = "maxpool"

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling

        context_length = batch_input_ids.shape[-1]
        cache_config = build_quantized_cache_config(
            args.quant_method,
            nbits=args.nbits,
            residual_length=args.quant_residual_length or output_max_len,
            device=args.quant_device,
            backend=args.quant_backend,
            q_group_size=args.q_group_size,
            axis_key=args.axis_key,
            axis_value=args.axis_value,
        )
        if cache_config is None:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=list(stop_token_ids)
            )
        else:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=list(stop_token_ids),
                cache_implementation="quantized",
                cache_config=cache_config,
            )

        # eval_batch_size is validated to be 1 at startup, so only sample 0 exists.
        batch_generations = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)

        torch.cuda.empty_cache()

        example = {}
        example["prompt"] = batch_prompts[0]
        example["input"] = batch_inputs[0]
        example["answers"] = batch_answers[0]
        example["pred"] = batch_generations[0]
        example["length"] = batch_lengths[0]

        fout.write(json.dumps(example) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=str2bool, default=True, help="")
    parser.add_argument("--output_attentions", type=str2bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=str2bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--quant_backend", type=str, default="hqq", choices=["hqq", "quanto"], help="Quantized cache backend.")
    parser.add_argument("--quant_device", type=str, default="cuda", help="Device used by the quantized cache backend.")
    parser.add_argument("--quant_residual_length", type=int, default=None, help="Full-precision residual window for quantized cache; defaults to max_new_tokens.")
    parser.add_argument("--q_group_size", type=int, default=64, help="Quantization group size.")
    parser.add_argument("--axis_key", type=int, default=None, choices=[0, 1], help="Override key quantization axis.")
    parser.add_argument("--axis_value", type=int, default=None, choices=[0, 1], help="Override value quantization axis.")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()

    if args.eval_batch_size != 1:
        raise ValueError("eval_batch_size != 1 is not supported yet: truncation and decoding only handle sample 0, so batching silently corrupts predictions - see issue #46.")

    ruler_supported_methods = ["fullkv", "snapkv", "pyramidkv", "h2o", "cam", "l2norm", "streamingllm"]
    if args.method.lower() not in ruler_supported_methods:
        raise ValueError(f"method {args.method!r} is not supported on RULER; supported methods: {ruler_supported_methods}")

    set_seed(args.seed)
    patch_quantized_cache(args.quant_method)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from pyramidkv.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()

    stop_token_ids = build_stop_token_ids(model, tokenizer)

    save_dir = args.save_dir
    max_capacity_prompts = args.max_capacity_prompts
    
    for context_length in context_length_list:
        for idx, dataset in enumerate(datasets):

            print(f"Working on context length {context_length}, max_capacity_prompts: {args.max_capacity_prompts}, dataset: {dataset} - {idx}/{len(datasets)}")
            args.context_length = context_length
            args.dataset = dataset
            args.data_file = f"data/RULER/{context_length}/{args.dataset}.jsonl"

            main(args)
