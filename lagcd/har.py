import torch
from torch import nn

class HARLayer(nn.Module):
    def __init__(self,d,heads,ff,dropout):
        super().__init__();self.base=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.streams=nn.ModuleList([nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True) for _ in range(4)])
        self.gate=nn.Linear(4*d+64,4)
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff,d))
    def forward(self,x,stream_sources,j,mask):
        q=x+self.base(x,x,x,key_padding_mask=~mask,need_weights=False)[0]
        outs=[]
        for stream,source in zip(self.streams,stream_sources):
            outs.append(stream(q,source,source,need_weights=False)[0])
        stacked=torch.stack(outs,dim=2) # B,N,4,D
        b,n,_,d=stacked.shape
        gate=torch.sigmoid(self.gate(torch.cat([stacked.reshape(b,n,4*d),j[:,None,:].expand(-1,n,-1)],-1)))
        fused=(stacked*gate.unsqueeze(-1)).sum(2)
        x=self.norm1(q+fused);x=self.norm2(x+self.ff(x))
        return x*mask.unsqueeze(-1),gate

class HierarchicalAttentionRouter(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg['d_model']
        self.layers=nn.ModuleList([HARLayer(d,cfg['heads'],cfg['ff_dim'],cfg['dropout']) for _ in range(cfg['har_layers'])])
        self.jur=nn.Embedding(cfg['num_jurisdictions'],64)
    def forward(self,x,sources,jurisdiction,mask):
        j=self.jur(jurisdiction); gates=[]
        for layer in self.layers:
            x,g=layer(x,sources,j,mask);gates.append(g)
        return x,torch.stack(gates)
