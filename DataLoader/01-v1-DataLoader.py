
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

CONTEXT_SIZE = 4 

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, strides):
        self.inputs_ids = []
        self.targets_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, strides):
            inputs = token_ids[i : i + max_length]
            target = token_ids[i+1: i+max_length+1]
            self.inputs_ids.append(torch.tensor(inputs))
            self.targets_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.inputs_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


            

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0) -> DataLoader:
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, strides=stride)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return data_loader





if __name__ == '__main__':
    if torch.backends.mps.is_available() == False:
        print(f'Pytorch acelerator for Apple Silicon: False')

    with open('./the-verdict.txt', 'r', encoding = 'utf-8') as f:
        text = f.read()

    data_loader = create_dataloader_v1(text)
    print('iok')







    