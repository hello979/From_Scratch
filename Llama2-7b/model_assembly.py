import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from RoPE import Theta,RoPE_apply
from RMS_Norm import RMSNorm


@dataclass
class ModelArgs:
    dim:int=4096
    n_layers:int=32 #how many times the block has been stacked before sending to RMS norm
    n_heads:int=32 #heads for the query
    n_kv_heads:Optional[int]=None # number of k and v heads
    vocab_size:int=-1 # will be set once we load the tokenizer
    multiple_of:int =256
    ffn_dim_multiplier: Optional[int]=None
    norm_eps:int=1e-5
    
    max_batch_size:int=32
    max_seq_length:int=2048

    device:str = None

class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()

        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)

        #Normalizing before self attention
        self.attention_norm=RMSNorm(args.dim,eps=args.norm_eps)
        #Normalizing befor the ffn
        self.ffn_norm=RMSNorm(args.dim,eps=args.norm_eps)

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor): 
        #start_pos is the strating pos of the token(for us we use only 1 token)
        #Check the architecture we apply self attention on the normalized version of the inputs
        #(b,seq_len,dim)+(b,seq_len,dim)-->(b,seq_len,dim)
        #Residual connections(h-->hidden )
        h=x+self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        output=h+self.feed_forward.forward(self.ffn_norm(h))

        return output

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    batch_size,seq_len,n_kv_heads,head_dim=x.shape
    if n_rep==1:
        return x
    else:
        return(
            x[:,:,:,None,:]
            .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
            .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        #we skipped ther parallelism as we dont have multiple GPUs(infact i dont have a single GPU T_T)
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q=args.n_heads
        self.n_rep=self.n_heads_q//self.n_kv_heads
        self.head_dim=args.dim//args.n_heads
        #The Linear is a dense network that simply does (xW^t+bias) if bias is set true
        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim, self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)


        self.cache_k=torch.zeros((args.max_batch_size,args.max_seq_length,self.n_kv_heads,self.head_dim))
        self.cache_v=torch.zeros((args.max_batch_size,args.max_seq_length,self.n_kv_heads,self.head_dim))

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len,_=x.shape #(B,1,dim)
        #Apply the wq,wk,wv matrices to querries keys and values
        #(B,1,dim)->(B,1,H_Q*Head_dim)
        xq=self.wq(x)
        #(B,1,dim)->(B,1,H_KV*Head_dim)
        xk=self.wk(x)
        xv=self.wv(x)

        #(B,1,H_Q*Head_dim)->(B,1,H_Q,Head_dim)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        #(B,1,H_KV*Head_dim)->(B,1,H_KV,Head_dim)
        xk=xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        #Does not change the shape of the tensor
        xq=RoPE_apply(xq,freqs_complex,device=x.device)
        xk=RoPE_apply(xk,freqs_complex,device=x.device)

        #K-V cache:
        #Append the new tokens in the Values and keys 
        #In forward() we recieve the latest token from the latest iterration and append it in the cache_k and cache_v
        self.cache_k[:batch_size,start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len]=xv
        #Retriev all the cached keys and values so far
        #(B,seq_len,H_KV,Head_dim)
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]
        #In Grouped Querry attention group of querry goes to a single kwy or value what we do is that we repeat the keys or values to the number of querries in a group and calculate as simple Attention mimicing gqa
        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(values,self.n_rep)
        #(B,1,H_Q,H_dim)-->(B,H_Q,1,H_dim)
        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)
        #(B,H_Q,1,Head_dim)@(B,H_Q,Head_dim,seq_len_KV)-->(B,H_Q,1,seq_lwn_KV)
        scores=torch.matmul(xq,keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=1).type_as(xq)

        #(B,H_Q,1,Seq_len)@(B,H_Q,Seq_len_KV,Head_dim)-->(B,H_Q,1,Head_dim)
        output=torch.matmul(scores,values)
        #(B,H_Q,1,Head_dim)-->(B,1,H_Q,Head_dim)-->(B,1,Head_dim)
        output=(output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output)
    
class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim=4 * args.dim
        hidden_dim=int(2*hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim=int(args.ffn_dim_multiplier*hidden_dim)
        #Round off the hidden _dim to the nearest multiple of the multiple_of parameter
        hidden_dim=args.multiple_of*((hidden_dim + args.multiple_of - 1)//args.multiple_of)

        self.w1=nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,args.dim,bias=False)
        self.w3=nn.Linear(args.dim,hidden_dim,bias=False)
    def forward(self,x:torch.Tensor):
        swish=F.silu(self.w1(x))
        x_v=self.w3(x)
        x=swish*x_v
        return self.w2(x)

class Transformer(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()

        assert args.vocab_size!=1, "Vocab size must be non zero"
        
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings=nn.Embedding(self.vocab_size,args.dim)

        self.layers=nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm=RMSNorm(args.dim,args.norm_eps)
        self.output=nn.Linear(args.dim,self.vocab_size,bias=False)
        self.freqs_complex=Theta(args.dim//args.n_heads,self.args.max_seq_length*2,device=self.args.device)

    def forward(self,tokens:torch.Tensor,start_pos:int):
        #we will pass only one token as kv cache is there
        #this model is ideal for inference not training, for training we need multiple tokens
        batch_size,seq_len=tokens.shape #(B,Seq_length)
        assert seq_len==1,"Only one token at a time can be processed"

        #(B,Seq_length)-->(B,Seq_length,dim)
        h=self.tok_embeddings(tokens)
        
        #Retriving the m and theta pairs corresponding to the positions [start_pos,start_pos+seq+length]
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        #Skeleton of the model
        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)
        h=self.norm(h)
        output=self.output(h).float()
        return output