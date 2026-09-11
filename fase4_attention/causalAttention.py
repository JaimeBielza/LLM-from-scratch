import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenght,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )

    def forward(self, x):
        num_tokens = x.shape
        keys = self.W_keys(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        atten_scores = queries @ keys.transpose(1, 2)
        atten_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        atten_weights = torch.softmax(atten_scores / keys.shape[-1]**0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vector = atten_weights @ values
        return context_vector


