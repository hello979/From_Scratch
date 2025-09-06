import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        #parameter gama(learnable)
        self.weights=nn.Parameter(torch.ones(dim))

    def norm(self,x:torch.Tensor):
        # (B,seq_length,dim)*(B,seq_length,1)=(B,seq_length,dim)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x:torch.Tensor):
        #(dim)*(B,seq_length,dim)-->(B,seq_length,dim)
        return self.weights*self.norm(x.float()).type_as(x)
