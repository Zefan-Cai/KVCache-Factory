import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(
    data: torch.Tensor, 
    title: str, 
    save_path=None
) -> None:
    """
    Helper function to plot a heatmap for a tensor of shape (seq_len, seq_len).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    seq_length = data.shape[0]

    _, ax = plt.subplots()
    ax.imshow(data, vmax=100)
    ax.set_xticks(np.arange(seq_length), labels=[])
    ax.set_yticks(np.arange(seq_length), labels=[])

    plt.title(title)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")

    plt.show()
