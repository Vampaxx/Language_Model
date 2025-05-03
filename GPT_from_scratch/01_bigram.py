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


torch.manual_seed(1337)
with open ("data\gpt_train.txt","r") as file:
    text    = file.read()


# All the unique characters that occur in this text 
chars       = sorted(list(set(text)))
vocab_size  = len(chars)

## creating mapping from characters to integers 
stoi    = {ch:i for i,ch in enumerate(chars)}
itos    = {i:ch for i,ch in enumerate(chars)}

encode  = lambda word: [stoi[i] for i in word]
#decode  = lambda integer: "".join(itos[int(i)] for i in integer)
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


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X,Y         = get_batch("train")
            logits,loss = model(X,Y)
            losses[k]   = loss.item()   
        out[split] = losses.mean()
    model.train()
    return out
        
## ------------------------------------------------------------------------ ## 
class BiGramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for next token from a lookup table 
        self.token_embedding_table  = nn.Embedding(vocab_size,vocab_size,device="cuda")
    def forward(self,idx,target=None):
        # index and target are both (B,T) tensor of integers
        logits  = self.token_embedding_table(idx)   # its arrange in the shape of (B,T,C)
        if target is None:
            loss    = None
        else:    
            B,T,C   = logits.shape
            logits  = logits.view(B*T,C)    # cross entropy input expectation is (minibatch,C)
            target  = target.view(B*T)      
            loss    = nn.functional.cross_entropy(logits,target)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        # idx is (B,T) array of indices in the current context. 
        for _ in range(max_new_tokens):
            logits,loss = self.forward(idx)                         # (B,T,C)
            logits      = logits[:,-1,:]                            # (B,C)
            Probs       = nn.functional.softmax(logits,dim=-1)      # (B,C)
            # sample from the distribution 
            idx_next    = torch.multinomial(Probs,num_samples=1)    # (B,1)
            # append sample index to running sequence 
            idx         = torch.cat([idx,idx_next],dim=1)           # (B,T+1)
        return idx
    
## ------------------------------------------------------------------------ ## 
model       = BiGramLanguageModel(vocab_size).to(device="cuda")
optimizer   = torch.optim.AdamW(model.parameters(),lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses  = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},val loss {losses['val']:.4f}")

    #sample a batch of data
    xb,yb = get_batch("train")
    #evaluate the loss
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1),dtype=torch.long,device="cuda:0")
print(decode(model.generate(context,max_new_tokens=500)[0]))












