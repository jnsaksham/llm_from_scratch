import torch.nn as nn
import torch

class Embeddings:
    def __init__(self, vocab_size=50257, output_dim=256, context_length=4):
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_length = context_length
        
        # layers
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.positional_embedding_layer = nn.Embedding(context_length, output_dim)

    def get_embeddings(self, inputs):
        token_embeddings = self.token_embedding_layer(inputs)
        positional_embeddings = self.positional_embedding_layer(torch.arange(self.context_length))
        return token_embeddings, positional_embeddings