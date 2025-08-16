# Sparse Mixture of Experts (MoE) Transformer

This repository implements a **Sparse Mixture of Experts (MoE)** module integrated into a Transformer architecture. It selectively routes input embeddings to a subset of experts, making the model efficient while maintaining high capacity.

---

## Overview

The **Mixture of Experts (MoE)** mechanism allows a neural network to conditionally route input tokens to different expert subnetworks. This enables scaling models without linearly increasing computation.

Key components:

1. **Expert Network**: Expands the embedding, applies non-linearity, and contracts it back.
2. **Noisy Top-K Router**: Determines which experts to use for each token.
3. **Sparse MoE Module**: Combines expert outputs weighted by the router.

---

## Formulas

### Top-K Gating
For an input embedding \( x \in \mathbb{R}^{d} \) and \( E \) experts:

1. Compute logits for each expert:  
\[
\text{logits} = W_r x
\]  
where \( W_r \in \mathbb{R}^{d \times E} \) is the router weight matrix.

2. Add noise for exploration:
\[
\text{noisy\_logits} = \text{logits} + \mathcal{N}(0, (\text{softplus}(W_n x))^2)
\]

3. Select top-k experts:
\[
\text{indices}, \text{top\_k\_logits} = \text{topk}(\text{noisy\_logits}, k)
\]

4. Mask non-selected experts to \(-\infty\) and apply softmax:
\[
\text{router\_output}_i = \frac{e^{\text{top\_k\_logits}_i}}{\sum_{j \in \text{top-k}} e^{\text{top\_k\_logits}_j}}
\]

> Note: \( e^{-\infty} = 0 \), ensuring non-selected experts contribute nothing.

---

## Code Structure

### Expert

```python
class Expert(nn.Module):
    """
    Expands input embedding 4x and contracts back.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed), # Expand
            nn.ReLU(),                      # Activation (ReLU or GeLU)
            nn.Linear(4 * n_embed, n_embed),# Contract
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
