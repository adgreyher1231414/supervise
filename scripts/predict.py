import argparse,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lagcd.model import LAGCD
from lagcd.data import ComplianceDataset,LABELS
from lagcd.engine import collate,move
from lagcd.utils import device_for
from lagcd.llm import LegalReasoner
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/demo/best.pt');p.add_argument('--input',required=True,help='JSON file containing a scenario');p.add_argument('--generate',action='store_true');args=p.parse_args()
c=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=c['config'];device=device_for(cfg['train']['device'])
model=LAGCD(cfg['model']).to(device);model.load_state_dict(c['model']);model.eval()
r=json.loads(Path(args.input).read_text(encoding='utf-8'))
if args.generate:
    llm=LegalReasoner(**cfg['api']);r['lcot_answers']=llm.steps(r.get('scenario',''),r.get('statutes',''))
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'input.jsonl';path.write_text(json.dumps(r)+'\n',encoding='utf-8')
    b=move(collate([ComplianceDataset(path,cfg)[0]]),device)
    with torch.no_grad(): o=model(b)
    result={'verdict':LABELS[int(o['logits'].argmax(-1).item())], 'risk':float(o['risk'].item()),'probabilities':o['probabilities'][0].tolist(),'temperature':float(o['temperature'])}
    if o['conflict']:
        result['conflict_detected']=bool(o['conflict']['conflicts'].any().item())
        result['needs_human_review']=bool(o['conflict']['resolution_logits'].argmax(-1).item()==2 and result['conflict_detected'])
    if args.generate:result['rationale']=llm.rationale(r.get('scenario',''),r.get('statutes',''),result['verdict'],r['lcot_answers'])
    print(json.dumps(result,indent=2))
