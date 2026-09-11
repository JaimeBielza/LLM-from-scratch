import torch
import sys, os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from fase2_dataloader.DataLoader import create_dataloader_v1
from fase0_utils.ExtraFunctions import get_text



if __name__ == '__main__':
    vocab_size = 50257
    output_dim = 256
    max_length = 4
    torch.manual_seed(123)
    embeddings_layer = torch.nn.Embedding(vocab_size, output_dim)
    poss_embeddings_layer = torch.nn.Embedding(max_length, output_dim)


    text = get_text()
    dataLoader = create_dataloader_v1(
        text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )

    data_iter = iter(dataLoader)
    inputs, target = next(data_iter)

    token_embeddings_layer = embeddings_layer(inputs)
    pos_embeddings = poss_embeddings_layer(torch.arange(max_length))

    inputs_embeddings = token_embeddings_layer + pos_embeddings
    print(f"inputs embeddings: \n{inputs_embeddings}")
