import torch

class SimplifiedSelfAttention:
    """
    Bahdanau attention mechanism for RNNs: https://arxiv.org/abs/1409.0473

    Before this, RNNs updated hidden state via the encoder after each word, and sent the final hidden state to the decoder.
    This led to a lot of expectations from decoder as ONLY the hidden state was passed.

    Bahdanau attention mechanism also ensures access to input words for the decoder at each step, along with the hidden state.
    The weightage associated with all these input words is interpreted as "attention"
    
    """

    def attention_scores(self, queries):
        attention_scores = queries @ queries.T
        return attention_scores

    # compute attention weights
    def attention_weights(self, attention_scores):
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return attention_weights

    # compute context vector: weighted same of weights and inputs
    def context_vector(self, queries, attention_weights):
        context = attention_weights @ queries
        return context

class SelfAttention:
    pass

class CrossAttention:
    pass

class MultiHeadAttention:
    pass