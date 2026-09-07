import time
import os
import re





if __name__ == '__main__':
    with open('./the-verdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    simbols = list(set(text))
    simbols.extend(['<|unk|>'])
    print('ok')