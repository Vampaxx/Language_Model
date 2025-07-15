import torch 
import torch.nn as nn 
from torch.nn import functional as F

from tqdm import tqdm




batch_size  = 32
block_size  = 8
device      = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval   = 300
learning_rate   = 1e-3 
max_iters       = 3000
eval_iter       = 200
n_emd           = 32 


torch.manual_seed(1337)
with open ("data\gpt_train.txt","r") as file:
    text    = file.read()


# All the unique characters that occur in this text 
chars       = sorted((set(text)))
vocab_size  = len(chars)

## creating mapping from characters to integers 
stoi    = {ch:i for i,ch in enumerate(chars)}
itos    = dict(enumerate(chars))
encode  = lambda word: [stoi[i] for i in word]
decode  = lambda integers: "".join(itos[int(i)] for i in integers)



## Let's encode entire dataset of file. 
data = torch.tensor(encode(text),dtype=torch.long,device="cuda")

## Lets split the data into train and val dataset 
n   = int(0.9 * len(data))
train_data  = data[:n]
val_data    = data[n:]


def get_batch(split):
    data    = train_data if split == "train" else val_data
    ix      = torch.randint(len(data)-block_size,(4,))
    x       = torch.stack([data[i   : i+block_size] for i in ix]).to(device="cuda:0")
    y       = torch.stack([data[i+1 : i+block_size+1] for i in ix]).to(device="cuda:0")
    return x,y 

