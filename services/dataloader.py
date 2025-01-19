import os

def load_txt(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return f.read()