import torch
from torch import nn

class MultiJurisdictionConflictResolver(nn.Module):
    def __init__(self,d,num_j,threshold=.3):
        super().__init__();self.threshold=threshold;self.jur=nn.Embedding(num_j,d)
        self.compat=nn.Linear(2*d,1)
        self.resolve=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,3))
        self.fuse=nn.Linear(d,d)
    def forward(self,norms,norm_j,mask):
        b,n,d=norms.shape
        a=norms[:,:,None,:].expand(-1,-1,n,-1)
        c=norms[:,None,:,:].expand(-1,n,-1,-1)
        # Symmetric compatibility from averaged ordered pair scores.
        logits=self.compat(torch.cat([a,c],-1)).squeeze(-1)
        compatibility=(torch.tanh(logits)+torch.tanh(logits.transpose(1,2)))*.5
        diff=(norm_j[:,:,None]!=norm_j[:,None,:])
        valid=mask[:,:,None]&mask[:,None,:]&diff
        conflict=valid&(compatibility < -self.threshold)
        weighted=norms*mask.unsqueeze(-1)
        pooled=weighted.sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        # Most conflicting pair when available, otherwise first valid pair; deterministic.
        scores=compatibility.masked_fill(~valid,2.)
        flat=scores.reshape(b,-1).argmin(-1); i=flat//n; k=flat%n
        row=torch.arange(b,device=norms.device)
        ni=norms[row,i];nk=norms[row,k]
        ji=self.jur(norm_j[row,i]);jk=self.jur(norm_j[row,k])
        res_logits=self.resolve(torch.cat([ni,nk,ji,jk],-1))
        has_pairs=valid.flatten(1).any(1)
        res_logits=torch.where(has_pairs[:,None],res_logits,torch.zeros_like(res_logits))
        out=self.fuse(pooled+0.5*(ji+jk)*has_pairs[:,None])
        return out,{'compatibility':compatibility,'conflicts':conflict,'resolution_logits':res_logits,'has_pairs':has_pairs}
