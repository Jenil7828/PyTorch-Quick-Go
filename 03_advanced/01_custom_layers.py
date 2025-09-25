"""
PyTorch Advanced: Custom Layers and Modules
==========================================

This file covers creating custom layers and modules:
- Custom nn.Module implementations
- Parameter handling and initialization
- Forward and backward pass customization
- Functional vs Module approaches
- Advanced layer types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np

class LinearWithCustomInit(nn.Module):
    """Custom linear layer with specialized initialization"""
    
    def __init__(self, in_features, out_features, bias=True, init_type='xavier'):
        super(LinearWithCustomInit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_type = init_type
        
        # Define parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters based on init_type"""
        if self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        elif self.init_type == 'orthogonal':
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.uniform_(self.weight, -0.1, 0.1)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        """String representation for the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, init_type={self.init_type}'

class MultiHeadAttention(nn.Module):
    """Custom multi-head attention implementation"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class ResidualConnection(nn.Module):
    """Residual connection with layer normalization"""
    
    def __init__(self, size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_fn):
        """Apply residual connection to sublayer function"""
        return x + self.dropout(sublayer_fn(self.norm(x)))

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SelfAttentionBlock(nn.Module):
    """Complete self-attention block with residual connections"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask)[0])
        
        # Feed-forward with residual connection
        x = self.residual2(x, self.feed_forward)
        
        return x

class SpectralNorm(nn.Module):
    """Spectral normalization wrapper"""
    
    def __init__(self, module, name='weight', n_power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        weight = getattr(module, name)
        height = weight.data.shape[0]
        width = weight.data.view(height, -1).shape[1]
        
        u = weight.new_empty(height).normal_(0, 1)
        v = weight.new_empty(width).normal_(0, 1)
        
        self.register_buffer(name + "_u", F.normalize(u, dim=0))
        self.register_buffer(name + "_v", F.normalize(v, dim=0))
        
        del module._parameters[name]
        self.register_parameter(name + "_orig", nn.Parameter(weight.data))
    
    def forward(self, *args, **kwargs):
        self._update_uv()
        return self.module.forward(*args, **kwargs)
    
    def _update_uv(self):
        weight = getattr(self, self.name + "_orig")
        u = getattr(self, self.name + "_u")
        v = getattr(self, self.name + "_v")
        
        height = weight.shape[0]
        weight_mat = weight.view(height, -1)
        
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0)
                u = F.normalize(torch.mv(weight_mat, v), dim=0)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight_normalized = weight / sigma
        setattr(self.module, self.name, weight_normalized)

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) implementation"""
    
    def __init__(self, input_size, hidden_size):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size * 2)
    
    def forward(self, x):
        # Split the output into two halves
        h = self.linear(x)
        h1, h2 = h.chunk(2, dim=-1)
        return h1 * torch.sigmoid(h2)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(batch_size, channels)
        # Excitation
        y = self.excitation(y).view(batch_size, channels, 1)
        # Scale
        return x * y.expand_as(x)

class ParametricReLU(nn.Module):
    """Parametric ReLU activation"""
    
    def __init__(self, num_parameters=1, init=0.25):
        super(ParametricReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
    
    def forward(self, x):
        return F.prelu(x, self.weight)

class LayerNorm1d(nn.Module):
    """1D Layer normalization"""
    
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # x shape: (batch_size, num_features, length)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.weight.unsqueeze(0).unsqueeze(2) * (x - mean) / (std + self.eps) + self.bias.unsqueeze(0).unsqueeze(2)

def test_custom_layers():
    """Test all custom layers"""
    print("=== Testing Custom Layers ===")
    
    batch_size, seq_len, d_model = 32, 100, 512
    
    # Test LinearWithCustomInit
    print("Testing LinearWithCustomInit...")
    linear_layer = LinearWithCustomInit(d_model, 256, init_type='xavier')
    x = torch.randn(batch_size, d_model)
    out = linear_layer(x)
    print(f"Linear output shape: {out.shape}")
    
    # Test MultiHeadAttention
    print("\nTesting MultiHeadAttention...")
    mha = MultiHeadAttention(d_model, num_heads=8)
    x = torch.randn(batch_size, seq_len, d_model)
    attn_out, attn_weights = mha(x, x, x)
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test PositionalEncoding
    print("\nTesting PositionalEncoding...")
    pos_enc = PositionalEncoding(d_model)
    x_with_pos = pos_enc(x)
    print(f"Positionally encoded shape: {x_with_pos.shape}")
    
    # Test SelfAttentionBlock
    print("\nTesting SelfAttentionBlock...")
    sa_block = SelfAttentionBlock(d_model, num_heads=8, d_ff=2048)
    sa_out = sa_block(x)
    print(f"Self-attention block output shape: {sa_out.shape}")
    
    # Test GatedLinearUnit
    print("\nTesting GatedLinearUnit...")
    glu = GatedLinearUnit(d_model, 256)
    x_1d = torch.randn(batch_size, d_model)
    glu_out = glu(x_1d)
    print(f"GLU output shape: {glu_out.shape}")
    
    # Test SqueezeExcitation
    print("\nTesting SqueezeExcitation...")
    se = SqueezeExcitation(channels=128)
    x_se = torch.randn(batch_size, 128, seq_len)
    se_out = se(x_se)
    print(f"Squeeze-Excitation output shape: {se_out.shape}")
    
    # Test ParametricReLU
    print("\nTesting ParametricReLU...")
    prelu = ParametricReLU(num_parameters=d_model)
    x_prelu = torch.randn(batch_size, d_model) - 1  # Some negative values
    prelu_out = prelu(x_prelu)
    print(f"Parametric ReLU output shape: {prelu_out.shape}")
    print(f"PReLU parameter: {prelu.weight[0].item():.4f}")
    
    return linear_layer, mha, sa_block

def complex_model_example():
    """Example of a complex model using custom layers"""
    print("\n=== Complex Model Example ===")
    
    class TransformerEncoder(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=1000):
            super(TransformerEncoder, self).__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            
            self.layers = nn.ModuleList([
                SelfAttentionBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ])
            
            self.norm = nn.LayerNorm(d_model)
            self.classifier = LinearWithCustomInit(d_model, 2, init_type='xavier')
        
        def forward(self, x):
            # Embedding and positional encoding
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            
            # Apply transformer layers
            for layer in self.layers:
                x = layer(x)
            
            # Classification
            x = self.norm(x)
            x = x.mean(dim=1)  # Global average pooling
            x = self.classifier(x)
            
            return x
    
    # Create and test the model
    model = TransformerEncoder(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024
    )
    
    # Test forward pass
    batch_size, seq_len = 16, 50
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    output = model(input_ids)
    
    print(f"Model input shape: {input_ids.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def custom_function_example():
    """Example of custom autograd function"""
    print("\n=== Custom Autograd Function ===")
    
    class SquareFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input ** 2
        
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            return grad_output * 2 * input
    
    # Use custom function
    x = torch.tensor(3.0, requires_grad=True)
    y = SquareFunction.apply(x)
    y.backward()
    
    print(f"x = {x.item()}")
    print(f"y = x² = {y.item()}")
    print(f"dy/dx = {x.grad.item()} (expected: 6)")
    
    return SquareFunction

def parameter_sharing_example():
    """Example of parameter sharing between layers"""
    print("\n=== Parameter Sharing Example ===")
    
    class TiedWeightsModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super(TiedWeightsModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.hidden = nn.Linear(hidden_size, hidden_size)
            
            # Output layer shares weights with embedding
            self.output = nn.Linear(hidden_size, vocab_size, bias=False)
            self.output.weight = self.embedding.weight  # Weight tying
        
        def forward(self, x):
            x = self.embedding(x)
            x = torch.relu(self.hidden(x))
            x = self.output(x)
            return x
    
    model = TiedWeightsModel(vocab_size=1000, hidden_size=128)
    
    # Verify weight sharing
    print(f"Embedding weight shape: {model.embedding.weight.shape}")
    print(f"Output weight shape: {model.output.weight.shape}")
    print(f"Weights are shared: {model.embedding.weight is model.output.weight}")
    
    # Count unique parameters
    unique_params = set()
    for param in model.parameters():
        unique_params.add(param.data_ptr())
    
    total_params = sum(p.numel() for p in model.parameters())
    unique_param_count = sum(p.numel() for p in unique_params)
    
    print(f"Total parameter references: {total_params:,}")
    print(f"Unique parameters: {len(unique_params)}")
    
    return model

def main():
    """Run all examples"""
    print("PyTorch Custom Layers Tutorial")
    print("=" * 40)
    
    test_custom_layers()
    complex_model_example()
    custom_function_example()
    parameter_sharing_example()
    
    print("\n" + "=" * 40)
    print("Tutorial completed successfully!")

if __name__ == "__main__":
    main()