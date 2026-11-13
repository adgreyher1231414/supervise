import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lagcd.model import LAGCD
from lagcd.data import ComplianceDataset
from lagcd.engine import evaluate
from lagcd.utils import device_for
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/demo/best.pt');p.add_argument('--test',default='data/demo/test.jsonl');p.add_argument('--out',default='runs/demo/metrics.json');args=p.parse_args()
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=checkpoint['config'];device=device_for(cfg['train']['device'])
model=LAGCD(cfg['model']).to(device);model.load_state_dict(checkpoint['model'])
result=evaluate(model,ComplianceDataset(args.test,cfg),cfg['train']['batch_size'],device)
Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
