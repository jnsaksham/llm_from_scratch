import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

def load_txt(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()
    

class DostoyevskyDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # sliding window for input output pairing
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length], dtype=torch.long))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = DostoyevskyDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader
    