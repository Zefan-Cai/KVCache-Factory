import os
import json
import random
import argparse
import subprocess
import datetime as _datetime

import numpy as np
from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyramidkv.quantization import build_quantized_cache_config, patch_quantized_cache
from pyramidkv.eval_utils import str2bool, build_stop_token_ids

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

# LongBench datasets that must NOT be wrapped with a chat template
# (few-shot / code completion tasks), matching official LongBench pred.py.
datasets_no_chat = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# model2maxlen = {
#     "Llama-2-7b-chat-hf": 3950,
#     "Llama-3-8B-Instruct": 7950,
#     "Meta-Llama-3-70B-Instruct": 7950,
#     "Meta-Llama-3-8B-Instruct-32k": 31500,
#     "Llama-2-7B-32K-Instruct": 31500,
#     "Mistral-7B-Instruct-v0.2": 31500,
#     "Mistral-7B-Instruct-v0.1": 31500,

# }

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

def build_chat_llama3(prompt):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

def write_run_meta(save_dir, args):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    meta = {
        "git_commit": git_commit,
        "args": vars(args),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "timestamp": _datetime.datetime.now().isoformat(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4, default=str)

# def build_prompt(prompt, dataset):

#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
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
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
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
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        use_llama3_chat = ("llama-3" in model_path or "llama3" in model_path) \
            and args.dataset not in datasets_no_chat

        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        prompt = batch_prompts[0]
        truncated = False
        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            truncated = True

        if use_llama3_chat:
            # Wrap the already-middle-truncated prompt, matching official LongBench pred.py.
            # add_special_tokens=False avoids a double BOS on top of <|begin_of_text|>.
            prompt = build_chat_llama3(prompt)
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=False).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
        elif truncated:
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        # # default to True
        # if args.method == "DynamicKV":
        #     args.output_attentions = True
        # else:
        #     args.output_attentions=False

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        
        if args.method != "FullKV":
            if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv", "think"]:
                window_sizes = 8
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            if args.method.lower() =='headkv':
                with open(args.head_path, 'r') as file:
                    head_list = json.loads(file.readline())
                head_score_list = [np.mean(l[1]) for l in head_list.items()]
                head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                total_attention = head_score_list.reshape(model.config.num_hidden_layers, model.config.num_attention_heads)
                total_pool_capacity = (args.max_capacity_prompts // args.head_beta) * model.config.num_hidden_layers * model.config.num_attention_heads
                min_num = (args.max_capacity_prompts - args.max_capacity_prompts // args.head_beta)
                head_capacity = torch.round(total_attention * total_pool_capacity + min_num).int()
                model.model.config.head_capacity = head_capacity    

            kernel_sizes = 7
            pooling = "maxpool"
            ratio = args.pruning_ratio
            recent_size = args.recent_size

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            if not isinstance(ratio, list):
                ratio = [ratio] * layers
            if not isinstance(recent_size, list):
                recent_size = [recent_size] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.merge = args.merge
                model.model.layers[i].self_attn.config.floor = args.floor
                model.model.layers[i].self_attn.config.ratio = ratio[i]
                model.model.layers[i].self_attn.config.recent_size = recent_size[i]
                model.model.layers[i].self_attn.config.kv_cache_granularity = args.kv_cache_granularity
                model.model.layers[i].self_attn.config.gqa_score_agg = args.gqa_score_agg
            

        context_length = batch_input_ids.shape[-1]
        eos_token_ids = list(stop_token_ids)
        if args.dataset == "samsum":
            # Official LongBench stops samsum generations at the first newline.
            newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
            if newline_token_id not in eos_token_ids:
                eos_token_ids.append(newline_token_id)
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
                eos_token_id=eos_token_ids
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
                eos_token_id=eos_token_ids,
                cache_implementation="quantized",
                cache_config=cache_config,
            )

        # eval_batch_size is validated to be 1 at startup, so only sample 0 exists.
        batch_generations = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)

        torch.cuda.empty_cache()

        example = {}

        example["prompt"] = batch_prompts[0]
        example["input"] = batch_inputs[0]
        example["context"] = batch_contexts[0]
        example["answers"] = batch_answerss[0]
        example["pred"] = batch_generations[0]
        example["length"] = batch_lengths[0]

        example["dataset"] = batch_datasets[0]
        example["language"] = batch_languages[0]
        example["all_classes"] = batch_all_classess[0]
        example["_id"] = batch__ids[0]

        fout.write(json.dumps(example) + "\n")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="", help="deprecated alias of --datasets, kept for backward compatibility.")
    parser.add_argument("--datasets", type=str, default="", help="comma-separated LongBench datasets to evaluate; defaults to the full list.")
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
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "auto"], help="torch dtype used to load the model.")
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--kv_cache_granularity", type=str, default="query_head", choices=["query_head", "kv_head"], help="Granularity of the compressed KV cache (see docs/gqa_cache_layout.md).")
    parser.add_argument("--gqa_score_agg", type=str, default="mean", choices=["mean", "max", "sum"], help="How per-query-head scores are aggregated per KV head when kv_cache_granularity=kv_head.")
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
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--head_path', type=str, default='./data/heads_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json', help='Path to head score (HeadKV)')
    parser.add_argument('--head_beta', type=float, default=1.01, help='hyper-parameter used on HeadKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")

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

    if args.method is None:
        raise ValueError("--method is required (e.g. FullKV, SnapKV, PyramidKV, H2O, StreamingLLM, CAM, L2Norm).")

    if args.method.lower() == "think" and args.attn_implementation != "eager":
        raise ValueError("method 'think' only patches the eager attention path; with --attn_implementation flash_attention_2/sdpa it silently runs stock HF attention. Use --attn_implementation eager.")

    if args.kv_cache_granularity == "kv_head" and args.method.lower() not in ["snapkv", "pyramidkv", "h2o", "streamingllm", "cam", "l2norm"]:
        raise ValueError(f"kv_cache_granularity='kv_head' is not supported for method {args.method!r}; supported methods: snapkv, pyramidkv, h2o, streamingllm, cam, l2norm.")

    datasets_arg = args.datasets or args.dataset
    if datasets_arg:
        selected_datasets = [d.strip() for d in datasets_arg.split(",") if d.strip()]
        invalid_datasets = [d for d in selected_datasets if d not in datasets]
        if invalid_datasets:
            raise ValueError(f"Unknown dataset(s) {invalid_datasets}; valid datasets: {datasets}")
    else:
        selected_datasets = list(datasets)

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

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
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

    model_name = args.model_path.lower().split("/")[-1]
    write_run_meta(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}"), args)

    for idx, dataset in enumerate(selected_datasets):

        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(selected_datasets)}")

        args.dataset = dataset

        args.data_file = f"data/LongBench/{args.dataset}.jsonl"

        main(args)
