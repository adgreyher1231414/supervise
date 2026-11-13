import torch
from torch import nn

class MeanTextEncoder(nn.Module):
    def __init__(self, vocab, d):
        super().__init__(); self.emb=nn.Embedding(vocab,d,padding_idx=0); self.ln=nn.LayerNorm(d)
    def forward(self, x):
        emb=self.emb(x); mask=(x!=0).unsqueeze(-1)
        return self.ln((emb*mask).sum(-2)/mask.sum(-2).clamp(min=1))

class StatutoryParser(nn.Module):
    def __init__(self,d,heads,layers,gnn_layers,ff,dropout):
        super().__init__()
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,heads,ff,dropout,batch_first=True,activation='gelu',norm_first=True),layers,enable_nested_tensor=False)
        self.gnn=nn.ModuleList([nn.Linear(d*2,d) for _ in range(gnn_layers)])
    def forward(self,nodes,mask,adj):
        x=self.transformer(nodes,src_key_padding_mask=~mask)
        a=adj*mask[:,None,:].to(adj.dtype)*mask[:,:,None].to(adj.dtype)
        a=a/a.sum(-1,keepdim=True).clamp(min=1)
        for layer in self.gnn: x=torch.nn.functional.gelu(layer(torch.cat([x,torch.bmm(a,x)],-1))) *mask.unsqueeze(-1)
        return x

class LegalKnowledgeEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg['d_model'];self.text=MeanTextEncoder(cfg['vocab_size'],d)
        self.parser=StatutoryParser(d,cfg['heads'],cfg['stat_layers'],cfg['gnn_layers'],cfg['ff_dim'],cfg['dropout'])
        # Entity and RAG features can be supplied via context and are projected to the common d space.
        self.context_proj=nn.ModuleList([nn.Linear(d,d) for _ in range(3)])
    def forward(self,b):
        nodes=self.parser(self.text(b['nodes']),b['node_mask'],b['adj'])
        ctx=self.text(b['context'])
        ctx=torch.stack([p(ctx[:,i]) for i,p in enumerate(self.context_proj)],dim=1)
        return nodes,ctx
