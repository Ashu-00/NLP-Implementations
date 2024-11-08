import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # to be set when tokenizer is loaded
    multiple: int = 256
    ffn_dim_nultiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # needed for kv-cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_freq(head_dim: int, seq_len: int, device: str, theta_0 : float = 10000):
    
    assert  head_dim %2 ==0, "Head_dim must be even" # to form pairs
    
    theta_power = torch.arange(0, head_dim/2 , 2).float()/head_dim
    theta = 1/(theta_0 ** (theta_power)).to(device)
    # theta formula = 10000**(-2(i-1)/d) , i =[1,2,..., d/2]
    
    
    m = torch.arange(seq_len, device=device)
    
    outer_product = torch.outer(m, theta).float() 
    # op =  [   [m1*th1 m1*th2 ...] 
    #           [m2*th1 m2*th2 ...] 
    #           ....
    #       ]
    
    
    freqs_complex = torch.polar(torch.ones_like(outer_product),outer_product)
    # (cos(m1*th1) + i*sin(m1*th1)) form
    
    return freqs_complex

def apply_rot_emb(x: torch.Tensor , freqs_complex: torch.Tensor , device: str ):
    # step1-> pair adjacent x(embed elements) and convert to complex
    # dim(x) = (B,seq_len,H,head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1 ,2))
    # dim (x_complex) = (B, seq_len, H, head_dim/2) -> no. of elements halved as complex no. has two comp.
    
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # to match dim of x_complex
    
    x_rotated = x_complex * freqs_complex # step2-> element wise multiplication 
    
    x_real = torch.view_as_real(x_rotated) # dim changes to (B,seq_len,H,head_dim/2 ,2) as all elements are real
    
    x_out = x_real.reshape(*x.shape) # dim = (B,seq_len,H,head_dim)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    if n_rep ==1 :
        return x
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:,:,:,None,:] # (batch_size, seq_len, n_kv_heads, 1, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim) # (batch_size, seq_len, n_q_heads, 1, head_dim)
    )
    


class RMSnorm(nn.Module):
    def __init__(self, dim: int, eps:float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps= eps
    
    def _norm(self, x:torch.Tensor):
        # to return x * 1/rms(x)
        # rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim= True) + self.eps)
    
    def forward(self, x:torch.tensor):
        # scaling of x/rms(x) by a weight
        return self.weight * self._norm(x)

class SelfAttention(nn.Module):
    def __init__(self , args : ModelArgs):
        super().__init__()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads == None else args.n_kv_heads #no. of heads for key and values
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads # no. of repetitions for k_v_heads to match q heads
        
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self,x: torch.Tensor, start_pos:int, freqs_complex:torch.Tensor ):
        batch_size,seq_len, _ = x.shape #(seq_len =1)&(_ = dim)
        #APPLY weights wq,wv,wk
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # old xq-> (Batch size, seq_len(=1) , H_q*head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q , self.head_dim)
        
        # old xk,xv -> (Batch size, seq_len(=1) , H_kv*head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads , self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads , self.head_dim)
        
        #apply rotary embeddings only to keys and queries
        xq = apply_rot_emb(xq, freqs_complex, x.device)
        xk = apply_rot_emb(xk, freqs_complex, x.device)
        
        #load new tokens in cache
        self.cache_k[:batch_size,start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len] = xv
        
        #load k,v from cache
        keys = self.cache_k[:batch_size, 0:seq_len+start_pos]
        values = self.cache_v[:batch_size, 0:seq_len+start_pos]
        
        # repeat heads of kv for grouping with q (kv_head<q_head)
        keys=repeat_kv(keys, self.n_rep)
        values=repeat_kv(values, self.n_rep)
        
        #ATTENTION MECHANISM
        
        # change dim (B,seq_len, H_q, head_dim)-> (B, H_q, seq_len, head_dim) for matmul
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        # softmax((Q.K_T).sqrt(head_dim))
        scores = (torch.matmul(xq, keys.transpose(2,3))/ torch.sqrt(self.head_dim))
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores,values)
        
        # concat heads (B , heads_q ,1 ,head_dim)-> (B, 1, dim)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # no change in dim in linear layer wo

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x
        
class EncoderBlock(nn.Module):
    def __init__(self,args: ModelArgs) :
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        
        self.attention = SelfAttention(args)
        self.feedforward = FeedForward(args)
        
        self.norm_att = RMSnorm(self.dim, args.norm_eps) #RMS before Self attention
        self.norm_ffn = RMSnorm(self.dim, args.norm_eps) #RMS beform ffn
    
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.norm_att(x), start_pos , freqs_complex)
        out = h + self.feedforward.forward(self.norm_ffn(h))
        
        return out
        

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab Size not set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSnorm(args.dim, eps=args.eps)
        
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.freq_complex = precompute_theta_pos_freq(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):

        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "One token at a time"

        # (B,seq_len)-> (B,seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve (m,theta) pairs corresponding to pos
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]
        
        for layer in self.layers:
            h =layer(h,start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
