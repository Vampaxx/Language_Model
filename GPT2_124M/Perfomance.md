- Before Initializtion 
    - loss = [10.84 - 6.54]
    - ms   = 27ms 
- After Initialization 
    - loss  = as above
    - ms    = 20ms

- Full precidion   - fp32
    - B= 4,T=1024
    - ms                = 6855.1550ms 
    - token per secod   = 450 tokens per second 

- by doing `torch.set_float32_matmul_precision('high)`
    - B= 4,T=1024
    - ms        = 6153.3651 ms      (+10.23% improvement)
    - tok/sec   =  665.6520 tok/sec (+47.77% improvement)
    
- By doing `Automatic Mixed Precision`
    - B= 4,t = 1024
    - ms        = 5745.4529 ms,     (+6.20% improvement)
    - tok/sec   = 712.9116 tok/sec  (+7.09% improvement)

- By doing `Flash Attention`
    - B = 4,T= 1024
    - ms        = 2309.8841 ms      (+59.800%)
    - tok/sec   = 1773.2492 tok/sec (+149.06%)

- By doing power of 2 
    - B = 4,T= 1024
    - ms        = 1992.7044 ms      (+13.72%)
    - tok/sec   = 2055.4981 tok/sec (+15.09%)


- lr = 6e-4
    - B = 4,T= 1024
    - ms        = 1117 ms      (+43.96%)
    - tok/sec   = 3601 tok/sec (+75.23%)