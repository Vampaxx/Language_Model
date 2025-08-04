import time 
import torch 
import tiktoken 
import torch.nn as nn 
from torch.nn import functional as F 
from dataclasses import dataclass
from functools import lru_cache



@dataclass
class GPTConfig:
    block_size:int  = 1024  # ==> block size
    vocab_size:int  = 50257 # ==> numbers of tokens ==> 50000 merges + 256 bytes token + 1 special token <|endoftext|>
    n_layer:int     = 12
    n_head:int      = 12
    n_emb:int       = 768   # ==> embedding dim


class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_emb % config.n_head ==0
        self.c_attn = nn.Linear(config.n_emb,3*config.n_emb)    # combined attention==> key,query,value projection for all heads,but in a batch
        self.c_proj = nn.Linear(config.n_emb,config.n_emb)      # output projection
        self.c_proj.NANOGPT_SCALE_INIT  = 1

        self.n_head = config.n_head
        self.n_emb  = config.n_emb

    def forward(self,x):
        B,T,C       = x.shape # batch_size, sequence length, embedding dim
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh                    = "number of heads",
        # hs                    = "head size"
        # C (number of channels)= nh * hs
        # e.g. in GPT-2 (124M)==> n_head    =12,
        #                         hs        =64, ==> nh * hs = C = 768 channels in the Transformer
        qkv     = self.c_attn(x)                                                    # B,T,3*n_emb
        q,k,v   = qkv.split(self.n_emb,dim=2)                                       # B,T,n_emb
        q       = q.view(B,T,self.n_head,self.n_emb//self.n_head).transpose(1,2)    # B, sequence_length(T), n_heads(n_h), head_size(hs) ==> B, n_h,T,hs
        k       = k.view(B,T,self.n_head,self.n_emb//self.n_head).transpose(1,2)    # B, sequence_length(T), n_heads(n_h), head_size(hs) ==> B, n_h,T,hs
        v       = v.view(B,T,self.n_head,self.n_emb//self.n_head).transpose(1,2)    # B, sequence_length(T), n_heads(n_h), head_size(hs) ==> B, n_h,T,hs
        y       = F.scaled_dot_product_attention(q,k,v,is_causal=True) #(B,n_head,T,head_size) # Flash attention==> its faster,cleaner,and scale better than Head object that created (reference link:-https://github.com/Vampaxx/Language_Model/blob/main/GPT_from_scratch/07_Adding_new_parameters.py )
        y       = y.transpose(1,2).contiguous().view(B,T,C)
        #output projection
        y       = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_emb,4*config.n_emb)
        self.gelu   = nn.GELU(approximate="tanh")   # there is no reason to use this approximation in nowdays, the time they develop this approximation they faced speed issue. thats why developed approximation
        self.c_proj = nn.Linear(config.n_emb * 4,config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT  = 1
    def forward(self,x):
        x   = self.c_fc(x)
        x   = self.gelu(x)
        x   = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1   = nn.LayerNorm(config.n_emb)
        self.attn   = CasualSelfAttention(config)
        self.ln_2   = nn.LayerNorm(config.n_emb)
        self.mlp    = MLP(config)

    def forward(self,x):
        x   = x + self.attn(self.ln_1(x))
        x   = x + self.mlp(self.ln_2(x))
        return x    
    
## ---------------------------------------------------------GPT--------------------------------------## 
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte     = nn.Embedding(config.vocab_size,config.n_emb),           # token embeding
            wpe     = nn.Embedding(config.block_size,config.n_emb),           # position embedding
            h       = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),    # self attention heads
            ln_f    = nn.LayerNorm(config.n_emb)
        ))
        self.lm_head= nn.Linear(config.n_emb,config.vocab_size,bias=False)    # lm_head is following be softmax, and bias not make any sence or improvement in learning.
        # The bias term in this case would just add a constant to each token’s logit — this doesn’t meaningfully improve learning,
        #-----------------------weight sharing scheme ---------------------------------# 
        self.transformer.wte.weight  = self.lm_head.weight
        # ----------------------Parameter Initialization ------------------------------#
        self.apply(self._init_weights) 

    def _init_weights(self,module):
        std = 0.02 
        if isinstance(module,nn.Linear):
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std *= (2* self.config.n_layer) ** -0.5  # In a block there is 2 residual connection per layer, so total n_layer * 2 residual connection total 
            # (2* self.config.n_layer) ** -0.5 ==> this means 1 / sqrt(total residual connection)
            torch.nn.init.normal_(module.weight,mean = 0.0,std = std)  # weight initialization function 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std = std) 

    def forward(self,idx,target=None):
        # shape of idx is (B,T)
        B,T     = idx.shape
        assert T<=self.config.block_size, f"cannot forward sequence of length {T},block_size is only {self.config.block_size}"
        pos     = torch.arange(0,T,dtype=torch.long,device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)                             # position embedding of shape (_,T,n_emb)
        tok_emb = self.transformer.wte(idx)                             # token embedding of shape    (B,T,n_emb)

        x       = tok_emb + pos_emb         # (B,T,n_emb)
        for block in self.transformer.h:
            x = block(x)
        #forward the final layerorm and classifier
        x       = self.transformer.ln_f(x)
        logits  = self.lm_head(x)           # (B,T,n_emb)

        ##------------------------------Adding Target and Loss---------------------- ##
        loss    = None
        if target is None:
            loss  = None
        elif target is not None:
            loss  = F.cross_entropy(input   = logits.view(-1,logits.size(-1)),        # cross entropy does not like multi-dimensional input, flatten out into 2D
                                    target  = target.view(-1),)
        return logits,loss

## ----------------------------------------END OF GPT CODE------------------------------------------------------------## 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The available device is {device}")


num_return_sequences  = 5
max_length            = 30


import tiktoken

@lru_cache(maxsize=1)
def model_init():
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    return model 

model = model_init()
## --------------------------------------------------Data Loading ----------------------------------------------------## 


class DataLoaderLite:
    def __init__(self,B,T):
        self.B  = B     # batch 
        self.T  = T     # sequence length 
        with open("data\gpt_train.txt","r") as f:
            text    = f.read()
        enc         = tiktoken.get_encoding("gpt2")
        tokens      = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded of {len(self.tokens)} tokens")
        print(f"1 Epoch = {len(self.tokens) // (B*T)} Batches of token")

        self.current_position = 0 

    def next_batch(self):
        B,T     = self.B,self.T
        buff    = self.tokens[self.current_position:self.current_position+B*T+1]
        x       = (buff[:-1]).view(B,T)     # input 
        y       = (buff[1:]).view(B,T)      # target 
        self.current_position   += B*T 
        if self.current_position + (B*T+1)> len(self.tokens):
            self.current_position = 0 
        return x,y

## --------------------------------------------------End of Data Loading ------------------------------ -------------## 
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4,T=1024)

## --------------------------------------------------------------------------------------------------------------------##
torch.set_float32_matmul_precision('high')   # float32 (FP32) >> tensorFloat32 (TF32)  
## --------------------------------------------------------------------------------------------------------------------##

model = GPT(GPTConfig())
model.eval()
model = model.to(device)

## Optimizer 
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    t0  = time.time()
    optimizer.zero_grad()
    ## -------------------------------------Automatic mixed precision--------------------------------------------------##
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        logits,loss = model(x,y)
        # import code; code.interact(local=locals()) 
         
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()    # makes the CPU wait until the GPU finishes all its scheduled work.
    t1  = time.time()
    dt = (t1 - t0) * 1000  # convert seconds to milliseconds
    token_per_second    = (train_loader.B * train_loader.T) / (t1 - t0) 
    print(f"Step : {i} loss: {loss.item():.4f}, dt: {dt:.4f} ms, tok/sec: {token_per_second:.4f}",)

