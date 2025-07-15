### parameters

- batch_size  = 4     # B 
- block_size  = 8     # T
- n_emd       = 32    # C 
- eval_interval   = 300
- learning_rate   = 1e-3 
- max_iters       = 5000 
- eval_iter       = 200


- Selt attention (single head)
    - training loss = 2.46
    - val loss      = 2.4876

- multihead self attention 
    - training loss = 2.39
    - val loss      = 2.38 

- Residula connection 
    - training loss = 2.34 
    - val loss      = 2.38

- Adding layer normalization at block
    - training loss = 2.32
    - val loss      = 2.33



batch_size  = 64     # B 
block_size  = 256    # T
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



    - training loss = 2.24
    - val loss      = 2.24

n_layers = 6,n_heads = 3
    - trainig loss = 2.