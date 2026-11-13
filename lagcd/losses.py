import torch
import torch.nn.functional as F

def soft_ece(probs,labels,n_bins=15,bandwidth=0.08):
    # Differentiable smooth-bin ECE surrogate; NOT exact hard-binned ECE.
    conf,pred=probs.max(-1);acc=(pred==labels).float().detach()
    centers=torch.linspace(0,1,n_bins,device=probs.device)
    w=torch.exp(-.5*((conf[:,None]-centers[None,:])/bandwidth)**2)
    weights=w.sum(0).clamp(min=1e-8)
    dif=((w*conf[:,None]).sum(0)-(w*acc[:,None]).sum(0))/weights
    return (dif.abs()*weights).sum()/weights.sum()

def objective(out,b,model,cfg):
    tr=cfg['train']; cls=F.cross_entropy(out['logits'],b['label'])
    risk_mask=b['has_risk'].bool()
    reg=F.mse_loss(out['risk'][risk_mask]/100,b['risk'][risk_mask]) if risk_mask.any() else cls.new_zeros(())
    calib=soft_ece(out['probabilities'],b['label'])
    # API generation is not differentiable. Use supervised rationale embedding alignment
    # as a transparent auxiliary surrogate, only where reference texts are present.
    refs=[i for i,t in enumerate(b['rationale']) if t.strip()]
    gen=cls.new_zeros(())
    if refs:
        from .data import encode_text
        ids=torch.tensor([encode_text(b['rationale'][i],model.cfg['max_context_tokens'],model.cfg['vocab_size']) for i in refs],device=cls.device)
        target=model.encoder.text(ids).detach()
        gen=(1-F.cosine_similarity(out['rationale_embedding'][refs],target,dim=-1)).mean()
    conflict=cls.new_zeros(())
    if out['conflict'] and 'resolution_logits' in out['conflict']:
        labels=b['conflict_label'];valid=(labels>=0)&out['conflict']['has_pairs']
        if valid.any(): conflict=F.cross_entropy(out['conflict']['resolution_logits'][valid],labels[valid])
    total=tr['lambda_cls']*cls+tr['lambda_reg']*reg+tr['lambda_gen']*gen+tr['lambda_calib']*calib+tr['lambda_conflict']*conflict
    return total,{k:float(v.detach()) for k,v in {'cls':cls,'risk':reg,'rationale_surrogate':gen,'calib_surrogate':calib,'conflict':conflict}.items()}
