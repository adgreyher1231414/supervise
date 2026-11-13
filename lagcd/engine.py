import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .losses import objective
from .metrics import metrics

def collate(rows):
    out={}
    for key in rows[0]:
        out[key]=torch.stack([r[key] for r in rows]) if isinstance(rows[0][key],torch.Tensor) else [r[key] for r in rows]
    return out

def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

def loader(ds,batch_size,shuffle=False):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate)

@torch.no_grad()
def evaluate(model,ds,batch_size,device):
    model.eval(); ys=[];probs=[];risks=[];risk_labels=[];available=[]
    for b in loader(ds,batch_size):
        b=move(b,device);o=model(b)
        ys.extend(b['label'].cpu().tolist());probs.extend(o['probabilities'].cpu().tolist())
        risks.extend(o['risk'].cpu().tolist());risk_labels.extend((100*b['risk']).cpu().tolist());available.extend(b['has_risk'].cpu().tolist())
    return metrics(ys,probs,risks,risk_labels,available)

def train(model,train_ds,val_ds,cfg,device,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    tr=cfg['train']; train_loader=loader(train_ds,tr['batch_size'],shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay'])
    steps=max(1,len(train_loader)*tr['epochs']);warmup=max(1,int(steps*tr['warmup_fraction']))
    def lr_scale(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
    best=-1;bad=0;history=[]
    for epoch in range(1,tr['epochs']+1):
        model.train();losses=[]
        for b in train_loader:
            b=move(b,device);opt.zero_grad(set_to_none=True)
            out=model(b);loss,_=objective(out,b,model,cfg)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step();losses.append(loss.item())
        score=evaluate(model,val_ds,tr['batch_size'],device)
        record={'epoch':epoch,'train_loss':float(sum(losses)/len(losses)),**score};history.append(record)
        print(f"epoch {epoch:02d} loss={record['train_loss']:.4f} val_f1={score['macro_f1']:.4f} ECE={score['ece']:.4f}",flush=True)
        if score['macro_f1']>best:
            best=score['macro_f1'];bad=0
            torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch},output_dir/'best.pt')
        else:
            bad+=1
            if bad>=tr['patience']:break
    (output_dir/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return history
