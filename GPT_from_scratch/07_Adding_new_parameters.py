import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from common.data_processing import get_batch     


batch_size  = 64     # B 
block_size  = 128    # T
n_emd       = 384    # C 
device      = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval   = 500
learning_rate   = 1e-3
max_iters       = 6000 
eval_iter       = 200
n_layers        = 6 
n_heads         = 6
dropout         = 0.2
vocab_size      = 65 



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


## -----------------Self Attention Head --------------- ## 
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key    = nn.Linear(n_emd,head_size,bias=False)   # key projection
        self.query  = nn.Linear(n_emd,head_size,bias=False)   # query projection
        self.value  = nn.Linear(n_emd,head_size,bias=False)   # value projection 
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout    = nn.Dropout(dropout)
    def forward(self,x):
        _,T,C   = x.shape
        k       = self.key(x)   # (B,T,head_size) ==> 4,8,16
        q       = self.query(x) # (B,T,head_size) ==> 4,8,16   
        ## compute attention score(affinities)
        wei     = q @ k.transpose(-2,-1) * C ** -0.5   # (B,T,c) @ (B,C,T) ==> (B,T,T)
        wei     = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei     = F.softmax(wei,dim=-1)
        wei     = self.dropout(wei)
        # perform the weighted aggregation of the value
        v       = self.value(x) # (B,T,head_size) ==> 4,8,16
        out     = wei @ v       # (B,T,T) @ (B,T,head_size) ==> (B,T,head_size) 
        return out 

## ----------------Multi head attention layer --------------------## 
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads      = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj       = nn.Linear(n_emd,n_emd) 
        self.dropout    = nn.Dropout(dropout)   
    def forward(self,x):
        out     = torch.cat([head(x) for head in self.heads],dim=-1)   # concatenate in the channel dimension 
        out     = self.dropout(self.proj(out))   # projection is the linear transformation of the out 
        return out 
    
## --------------FeedForward Layer ------------------------------##
class FeedForward(nn.Module):
    def __init__(self,n_emd):
        super().__init__()
        self.net    = nn.Sequential(
            nn.Linear(n_emd, 4 * n_emd), # from Paper 'attention all you need'
            nn.ReLU(),   
            nn.Linear(4 * n_emd,n_emd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

## --------------Block: Residual connection----------------------## 
class Block(nn.Module):
    def __init__(self,n_emd,n_heads):
        super().__init__()
        head_size                   = n_emd // n_heads 
        self.self_attention_head    = MultiHeadAttention(n_heads,head_size)
        self.ffwd                   = FeedForward(n_emd)
        self.ln1                    = nn.LayerNorm(n_emd)
        self.ln2                    = nn.LayerNorm(n_emd)
    def forward(self,x):
        x   = self.self_attention_head(self.ln1(x))
        x   = self.ffwd(self.ln2(x))
        return x 
## --------------Language model ---------------------------------## 

class BiGramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for next token from a lookup table 
        self.token_embedding_table      = nn.Embedding(vocab_size,n_emd)
        self.position_embedding_table   = nn.Embedding(block_size,n_emd)
        self.block                      = nn.Sequential(*[Block(n_emd,n_heads) for _ in range(n_layers)])
        self.ln_f                         = nn.LayerNorm(n_emd) # final layer normalization
        self.lm_head                    = nn.Linear(n_emd,vocab_size)       
        

    def forward(self,idx,target=None):
        B,T         = idx.shape 
        # index and target are both (B,T) tensor of integers
        tok_emb     = self.token_embedding_table(idx)   # its arrange in the shape of ==================================================================>> (B,T,C)
        pos_emb     = self.position_embedding_table(torch.arange(T,device=device)) # This embdding gives the idea of where word is belong in a sentence=>> (T,C) 
        x           = tok_emb + pos_emb                 # combined representation of token and its position.======Broadcasting apply=>> (B,T,C) + (T,C) == (B,T,C) 
        x           = self.block(x)                     # 
        x           = self.ln_f(x)
        logits      = self.lm_head(x)                   # for getting token_emb to logits we need linear layer,shape ===================================>> (B,T,vocab_size) 

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
            idx_cond    = idx[:,-block_size:]                   # crop idx to last block_size tokens
            logits,loss = self.forward(idx_cond)                # Get the prediction ==>  (B,T,C)
            logits      = logits[:,-1,:]                        # focus only on last time step  ==>  (B,C)
            probs       = nn.functional.softmax(logits,dim=-1)  # (B,C)
            # sample from the distribution 
            idx_next    = torch.multinomial(probs,num_samples=1)    # (B,1)
            # append sample index to running sequence 
            idx         = torch.cat([idx,idx_next],dim=1)           # (B,T+1)
        return idx




model   = BiGramLanguageModel()
model   = model.to('cuda:0')


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