import os
from time import time
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()} 

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([,.!:;"()\'])', r'\1', text)
        return text


#Lo probamos
if __name__ == '__main__':
    with open("./the-verdict.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_text = re.split(r'([,.:;?_!()"\']|--|\s)', text)
    strip_text = [item.strip() for item in split_text if item.strip()]
    all_words = sorted(set(strip_text))
    all_words.extend(['<|endoffile|>', '<|unk|>'])
    vocab_size = len(all_words)
    vocab = {
        token:integer for integer, token in enumerate(all_words)
    }

    text_1 = "Hello world, we are doing some things"
    text_2 = "The desultory life of the Riviera"
    text = "<|endoftext|>".join((text_1, text_2))

    tokenizer = SimpleTokenizerV1(vocab=vocab)
    text_encode = tokenizer.encode(text)
    print(text_encode)
    text_decode = tokenizer.decode(text_encode)
    print('-'*60)
    print(text_decode)
