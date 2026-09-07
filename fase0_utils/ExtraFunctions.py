

def get_text():
    with open('the-verdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

if __name__ == '__main__':
    text = get_text()