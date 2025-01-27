import torch
import torch.nn as nn

class SimplifiedSelfAttention:
    """
    Bahdanau attention mechanism for RNNs: https://arxiv.org/abs/1409.0473

    Before this, RNNs updated hidden state via the encoder after each word, and sent the final hidden state to the decoder.
    This led to a lot of expectations from decoder as ONLY the hidden state was passed.

    Bahdanau attention mechanism also ensures access to input words for the decoder at each step, along with the hidden state.
    The weightage associated with all these input words is interpreted as "attention"
    
    """
    def attention_scores(self, inputs):
        attention_scores = inputs @ inputs.T
        return attention_scores

    # compute attention weights
    def attention_weights(self, attention_scores):
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return attention_weights

    # compute context vector: weighted same of weights and inputs
    def context_vectors(self, inputs, attention_weights):
        context = attention_weights @ inputs
        return context

class SelfAttention(nn.Module):
    """
    Note that simplified self attention just focuses on semantic similarity between words. The context is derived from
    simple dot product of fixed embedding vectors.

    To encode overall context of the sentence, novel relationships are formed between words specific to the usage.
    It's important to adjust attention weights with this context. Hence, it's better to use trainable weights, 
    while still focusing on the underlying attention mechanism.

    Self attention solves that using Key, Query and Value matrices (Wk, Wq, Wv).

    Note that the implementation is on one input and not a batch of inputs. Causal attention module implements batched input
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        
        # Same as above but with optimized weight initialization for qkv training
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = self.attention_scores(queries, keys)
        attention_weights = self.attention_weights(attention_scores, keys)
        context_vector = self.context_vectors(attention_weights, values)

        return context_vector
    
    def attention_scores(self, queries, keys):
        attention_scores = queries @ keys.T
        return attention_scores
    
    def attention_weights(self, attention_scores, keys):
        d_k = keys.shape[-1]

        # scale softmax tor reduce the values. It avoids peaky behaviour of softmax if values are large
        attention_scores = attention_scores/(d_k ** 0.5)

        # softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        return attention_weights
    
    def context_vectors(self, attention_weights, values):
        context = attention_weights @ values
        return context

class CausalAttention(nn.Module):
    """
    Implemented on a batch
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = self.attention_scores(queries, keys)
        
        masked_weights = self.apply_causal_attention(attention_scores, keys, num_tokens)

        dropout_weights = self.apply_dropout(masked_weights)        

        context_vector = dropout_weights @ values

        return context_vector
    
    def attention_scores(self, queries, keys):
        attention_scores = queries @ keys.transpose(1, 2)   # only interested in num_tokens and d_in. And not b. so no need to transpose along 0
        return attention_scores
    
    def attention_weights(self, attention_scores, keys):
        d_k = keys.shape[-1]

        # scale softmax tor reduce the values. It avoids peaky behaviour of softmax if values are large
        attention_scores = attention_scores/(d_k ** 0.5)

        # softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        return attention_weights
    
    def apply_dropout(self, weights):
        torch.manual.seed(42)
        dropout_weights = self.dropout(weights)
        return dropout_weights
        
    def causal_mask(self, attention_scores):
        context_length = attention_scores.shape[0]
        mask = torch.tril(torch.ones(context_length, context_length))
        return mask
    
    def apply_causal_attention_leakage(self, attention_scores, keys):
        """
        Leaky implementation in which attention_weights are computed before applying softmax. So causal mask will have leakage
        from what it's not supposed to see.
        This is just for demonstration and highlighting the issue.
        """
        mask = self.causal_mask(attention_scores)
        attention_weights = self.attention_weights(attention_scores, keys)
        masked_weights = attention_weights * mask
        row_sums = masked_weights.sum(dim=1, keepdim=True)
        masked_weights_norm = masked_weights / row_sums
        return masked_weights_norm
        
    def apply_causal_attention(self, attention_scores, keys, num_tokens):
        """
        Tackles the leakage issue of original causal attention method

        First builds an upper triangle infinity mask on attention scores to cancel the impact of future tokens.
        It then applies softmax to compute attention weights.
        """
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        masked_weights = self.attention_weights(attention_scores, keys)
        return masked_weights


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Actual (parallel) implementation of multiheads.

        In GPT like models, usually d_in = d_out
        d_out = desired output dimension. Note that this is not the dimension of a single head. Single head's dimension is
        calcuated by d_out // num_heads
        """

        super().__init__()
        assert (d_out % num_heads == 0), \
               "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads    # reduce the proj dim to match desired output dim

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Since d_out is the final output dimension, we unroll d_out into num_heads * head_dim.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # currently it's grouped by num_tokens. We want to group it by num_heads
        # (b, num_tokens, self.num_heads, self.head_dim) -> (b, self.num_heads, num_tokens, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = keys.transpose(1, 2)
        values = keys.transpose(1, 2)

        # compute causal self attention 
        attention_scores = self.attention_scores(queries, keys)

        masked_scores = self.apply_mask(attention_scores)

        # compute attention weights
        attention_weights = self.attention_weights(masked_scores, keys)
        attention_weights = self.dropout(attention_weights)

        # dimension till now are (b, num_heads, num_tokens, head_dim)
        # needs to be rolled back up to (b, num_tokens, d_out). This is done in 2 steps

        # step 1: swap num_heads and num_tokens
        context_vector = (attention_weights @ values).transpose(1,2)

        # step 2: roll up last 2 into 1
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        context_vector = self.out_proj(context_vector)  # optional

        return context_vector
    
    def apply_mask(self, attention_scores, num_tokens):
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)
        return attention_scores

    def attention_scores(self, queries, keys):
        attention_scores = queries @ keys.transpose(2, 3)   # last 2 dimensions contain the actual 2-D keys. First 2 are batch and num_heads
        return attention_scores
    
    def attention_weights(self, attention_scores, keys):
        d_k = keys.shape[-1]

        # scale softmax tor reduce the values. It avoids peaky behaviour of softmax if values are large
        attention_scores = attention_scores/(d_k ** 0.5)

        # softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        return attention_weights
        

