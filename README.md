# Sparse Mixture of Experts (MoE) Transformer

A PyTorch implementation of a Transformer architecture enhanced with Sparse Mixture of Experts (MoE) layers for improved model capacity and efficiency.

## 🚀 Overview

This repository implements a Sparse Mixture of Experts system that can be integrated into transformer architectures. The MoE approach allows the model to scale capacity while keeping computational costs manageable by activating only a subset of expert networks for each input token.

## 🏗️ Architecture

### Key Components

- **Expert Networks**: Feed-forward networks that expand input embeddings to 4x dimension before contracting back
- **Noisy Top-K Router**: Intelligent routing mechanism with learnable noise for better expert selection
- **Sparse MoE Layer**: Combines multiple experts with sparse activation patterns

### Architecture Diagram

```
Input → Router → Top-K Selection → Expert Networks → Weighted Combination → Output
   ↓        ↓                           ↓
Embeddings  Gating    Expert1, Expert2, ...    Final Output
           Weights    (only top-k activated)
```

## 📋 Implementation

### Expert Network

Each expert is a simple feed-forward network that expands the input to 4x its dimension:

```python
class Expert(nn.Module):
  '''
  The expert can be seen as an neural network that expands the input embedding into 4 times it's dimention and re contracts it back to it's original dimention
  '''
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4* n_embed), # take input and expand to 4 times the input dimenstion
        nn.ReLU(), #we can also use GeLU
        nn.Linear(4 * n_embed, n_embed), # contract it back to original dimension
        nn.Dropout(dropout),
    )

  def forward(self,x):
    return self.net(x)
```

### Noisy Top-K Router

The router determines which experts to activate for each token:

```python
class NoisyTopKRouter(nn.Module):
      def __init__(self, n_embed, num_experts, top_k):

        super(NoisyTopKRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

      def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices
```

### Sparse MoE Layer

The main MoE layer that combines routing and expert computation:

```python
class SparseMoE(nn.Module):
  def __init__(self,n_embed, num_experts, top_k):
    super(SparseMoE,self).__init__()
    self.router = NoisyTopKRouter(n_embed, num_experts, top_k)
    self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
    self.top_k = top_k

  def forward(self, x):
    gating_output, indices = self.router(x)
    final_output = torch.zeros_like(x)

    flat_x = x.view(-1, x.size(-1))
    flat_gating_output = gating_output.view(-1, gating_output.size(-1))

    for i,expert in enumerate(self.experts):
      expert_mask = (indices == i).any(dim=-1)
      flat_mask = expert_mask.view(-1)

      if flat_mask.any():
        expert_input = flat_x[flat_mask]
        expert_output = expert(expert_input)

        gating_scores = flat_gating_output[flat_mask,i].unsqueeze(1)
        weighted_output = expert_output * gating_scores

        final_output[expert_mask] += weighted_output.squeeze(1)

    return final_output
```

## ✨ Features

- **Sparse Activation**: Only top-k experts are activated per token, reducing computational overhead
- **Learnable Noise**: Router includes trainable noise for better load balancing across experts
- **Flexible Architecture**: Easy to integrate into existing transformer models
- **Scalable Design**: Add more experts without proportional increase in computation

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/moe-transformer.git
cd moe-transformer

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install numpy
```

## 📖 Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from moe import SparseMoE

# Initialize MoE layer
n_embed = 512
num_experts = 8
top_k = 2

moe_layer = SparseMoE(n_embed, num_experts, top_k)

# Forward pass
batch_size, seq_len = 4, 128
x = torch.randn(batch_size, seq_len, n_embed)
output = moe_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### Integration with Transformer

```python
class TransformerBlockWithMoE(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.attention = MultiHeadAttention(n_embed, n_head)
        self.moe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        # Attention block
        x = x + self.attention(self.ln1(x))
        # MoE block
        x = x + self.moe(self.ln2(x))
        return x
```

## 🔧 Configuration

### Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_embed` | Embedding dimension | 512 | 256-2048 |
| `num_experts` | Number of expert networks | 8 | 4-32 |
| `top_k` | Number of experts to activate | 2 | 1-4 |
| `dropout` | Dropout rate in experts | 0.1 | 0.0-0.3 |

### Expert Configuration

Each expert uses a 4x expansion ratio by default. You can modify this in the `Expert` class:

```python
# For different expansion ratios
nn.Linear(n_embed, expansion_ratio * n_embed)
```

## 📊 Performance

### Benefits of MoE

- **Increased Capacity**: More parameters without proportional compute increase
- **Specialization**: Each expert can specialize in different types of inputs
- **Efficiency**: Only a fraction of the model is active per token

### Computational Complexity

- **Traditional FFN**: O(d² × sequence_length)
- **MoE Layer**: O(k × d² × sequence_length / num_experts)

Where k is the top_k parameter.

## 🧪 Experiments

### Load Balancing

Monitor expert utilization to ensure balanced load:

```python
def analyze_expert_usage(moe_layer, dataloader):
    expert_counts = torch.zeros(moe_layer.router.topkroute_linear.out_features)
    
    for batch in dataloader:
        _, indices = moe_layer.router(batch)
        for idx in indices.flatten():
            expert_counts[idx] += 1
    
    return expert_counts
```

### Training Tips

1. **Auxiliary Loss**: Add load balancing loss to encourage uniform expert usage
2. **Learning Rate**: Use lower learning rates for router parameters
3. **Initialization**: Careful initialization of router weights is important

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```


⭐ **Star this repository if you found it helpful!**
