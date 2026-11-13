import torch
from torch import nn
from .encoding import LegalKnowledgeEncoder
from .lcot import LegalChainOfThought
from .har import HierarchicalAttentionRouter
from .mcr import MultiJurisdictionConflictResolver
from .uac import UncertaintyAwareCalibrator

class LAGCD(nn.Module):
    def __init__(self,cfg,ablate=()):
        super().__init__(); self.cfg=cfg; self.ablate=set(ablate);d=cfg['d_model']
        self.encoder=LegalKnowledgeEncoder(cfg);self.lcot=LegalChainOfThought(d)
        self.har=HierarchicalAttentionRouter(cfg)
        self.mcr=MultiJurisdictionConflictResolver(d,cfg['num_jurisdictions'],cfg['conflict_threshold'])
        self.uac=UncertaintyAwareCalibrator(d,cfg['num_classes'])
        self.rationale_head=nn.Linear(3*d,d)
        self.fallback_attn=nn.MultiheadAttention(d,cfg['heads'],batch_first=True)
    def forward(self,b):
        nodes,ctx=self.encoder(b); d=nodes.shape[-1]
        if 'lcot' in self.ablate: r=torch.zeros_like(nodes[:,0]);states=torch.zeros(nodes.shape[0],5,d,device=nodes.device)
        else: r,states=self.lcot(self.encoder.text(b['answers']))
        # Four sources: statutory, case law, policy, entity.
        sources=[nodes,ctx[:,0:1],ctx[:,1:2],ctx[:,2:3]]
        x=torch.cat([nodes,ctx,states],dim=1)
        mask=torch.cat([b['node_mask'],torch.ones(x.shape[0],3+5,device=x.device,dtype=torch.bool)],dim=1)
        if 'har' in self.ablate:
            x=self.fallback_attn(x,x,x,key_padding_mask=~mask,need_weights=False)[0];gates=None
        else: x,gates=self.har(x,sources,b['jurisdiction'],mask)
        # First valid provision node is the CLS approximation.
        pooled=x[:,0]
        if 'mcr' in self.ablate:
            z_mcr=torch.zeros_like(pooled); info={}
        else:
            norm_embeddings=self.encoder.text(b['norms'])
            z_mcr,info=self.mcr(norm_embeddings,b['norm_jurisdictions'],b['norm_mask'])
        z=torch.cat([r,pooled,z_mcr],-1)
        out=self.uac(z)
        if 'uac' in self.ablate:
            out['logits']=out['raw_logits'];out['probabilities']=out['logits'].softmax(-1)
        out.update({'features':z,'rationale_embedding':self.rationale_head(z),'gates':gates,'conflict':info})
        return out
