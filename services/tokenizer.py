import re
import importlib
import tiktoken

class VocabBuilder:
    
    def add_special_tokens(self, vocab):
        special_tokens = ["<|endoftext|>", "<|unk|>"]
        vocab_size = len(vocab.items())
        for i, tok in enumerate(special_tokens):
            vocab[tok] = vocab_size + i - 1
        return vocab
    
    def create_word_vocab(self, text):
        # Convert text into tokens. Split special characters using re
        words = re.split(r'[.,:;?!_"()\']|--|\s', text)

        # remove spaces
        words = [item.strip() for item in words if item.strip()]

        # map tokens to token IDs
        words = sorted(set(words))

        vocab = {}
        for i, word in enumerate(words):
            vocab[word] = i

        vocab = self.add_special_tokens(vocab)
        
        return vocab
    
    def create_character_vocab(self, text):
        chars = list(text)
        
        # Don't remove white spaces as they're important for char tokenization
        chars = sorted(set(chars))
        
        vocab = {char:i for i, char in enumerate(chars)}
        
        vocab = self.add_special_tokens(vocab)

        return vocab

class WordTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        text = re.split(r'[.,:;?!_"()\']|--|\s', text)
        tokens = [item.strip() for item in text if item.strip()]
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in tokens]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[id] for id in token_ids])

        # replace spaces before the special characters
        text = re.sub(r'\s+([.,:;?!_"()\'])', r'\1', text)
        return text
    
class CharacterTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        chars = list(text)
        
        # Don't remove white spaces as they're important for char tokenization
        tokens = [item if item in self.str_to_int else "<|unk|>" for item in chars]
        
        token_ids = [self.str_to_int[token] for token in tokens]
        
        return token_ids

    def decode(self, token_ids):
        text = "".join(self.int_to_str[id] for id in token_ids)
        return text