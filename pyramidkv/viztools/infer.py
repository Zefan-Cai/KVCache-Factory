import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def infer_attention(
    model: AutoTokenizer, 
    tokenizer: AutoModelForCausalLM, 
    prompt: str, 
    device: str = "cpu", 
    amp: float = 10000.0
) -> list[torch.Tensor]:
    """
    Retrieve attention weights and return them as a list of tensors.
    
    Args:
        prompt (str): The input text to analyze.
        amp (float, optional): Amplification factor for the attention weights. Default to 10000.

    Returns:
        attentions (list[torch.Tensor, ...]): A list containing `#model_layers` tensors, each of shape (num_heads, seq_len, seq_len)
    """
    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attentions = model(input_ids, output_attentions=True).attentions

    return [attention.squeeze(0).detach().cpu() * amp for attention in attentions]
