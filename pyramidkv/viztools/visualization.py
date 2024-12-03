import os
import torch
from .utils import plot_heatmap


OUTPUT_DIR = "../obs"

def plot_attention_heatmap(attentions, head_ids=None, layer_ids=None, save_dir=""):
    """
    Visualizes attention weights as a heatmap.

    Args:
        attentions (torch.Tensor): Attention weights tensor of shape (num_heads, seq_len, seq_len).
        head_ids (list[int], optional): List of head indices to visualize. If None, all heads are averaged.
    """
    assert layer_ids is not None, "Please provide the layer_ids to visualize."
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if not head_ids:
            data = torch.mean(attention, dim=0).numpy()
            save_path = os.path.join(OUTPUT_DIR, save_dir, f"layer{layer_id}.jpg") if save_dir else None
            plot_heatmap(data, f'Average Attention Map: Layer {layer_id}', save_path)
        else:
            for head_id in head_ids:
                data = attention[head_id].numpy()
                save_path = os.path.join(OUTPUT_DIR, save_dir, f"layer{layer_id}_head{head_id}.jpg") if save_dir else None
                plot_heatmap(data, f'Attention Map: Layer {layer_id} Head {head_id}', save_path)
