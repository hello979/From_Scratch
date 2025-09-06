import torch
def Theta(head_dim:int,seq_len:int,device:str,theta:float=10000.00):
    #According to the paper the dimension of the embedding must be even
    assert head_dim%2==0,"Head dimension must be even"
    #Bulding the theta params
    #fromula theta_i=10000^(-2i/dim) where theta : [0,1,2.....dim_size/2]
    #shape:(head_dim/2)
    theta_numerator=torch.arange(0,head_dim,2).float()
    theta=1.0/(theta**(theta_numerator/head_dim)).to(device)
    #building the positions the "m" params
    m=torch.arange(seq_len,device=device)
    #multiply each theta to each m param
    freqs=torch.outer(m,theta).float()
    #complex numbers in polar form
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex

def RoPE_apply(x:torch.Tensor,freqs_complex:torch.Tensor,device:str):
    #transformation for applying RoPE:
    #(B,seq_length,H,head_dim)-->(B,seq_length,H,head_dim/2)
    #grouping consecitive vectors together
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    #freqs_complex:(seq_length,head_dim/2) so less dim than our x_complex
    #(seq_length,head_dim/2)--->(1,seq_length,1,head_dim/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    #(B,seq_length,H,head_dim/2)*(1,seq_length,1,head_dim/2)-->(B,seq_length,H,head_dim/2)
    x_rotated=x_complex*freqs_complex
    #(B,seq_length,H,head_dim/2)-->(B,seq_length,H,head_dim/2,2)
    x_out=torch.view_as_real(x_rotated)
    #(B,seq_length,H,head_dim/2,2)-->(B,seq_length,H,head_dim)
    x_out=x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
