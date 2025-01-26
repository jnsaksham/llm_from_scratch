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
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        # self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        
        # Same as above but with optimized weight initialization for qkv training
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        keys = inputs @ self.W_key
        queries = inputs @ self.W_query
        values = inputs @ self.W_value

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

class CrossAttention:
    pass

class MultiHeadAttention:
    pass