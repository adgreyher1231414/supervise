import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lagcd.data import ComplianceDataset
from lagcd.engine import train
from lagcd.model import LAGCD
from lagcd.utils import config,seed_all,device_for,count_parameters
p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--data',default='data/demo');p.add_argument('--out',default='runs/demo');p.add_argument('--ablate',nargs='*',choices=['lcot','har','mcr','uac'],default=[]);args=p.parse_args()
cfg=config(args.config);seed_all(cfg['seed']);device=device_for(cfg['train']['device']);model=LAGCD(cfg['model'],args.ablate).to(device)
print('Trainable parameters:',count_parameters(model),'Device:',device)
train(model,ComplianceDataset(Path(args.data)/'train.jsonl',cfg),ComplianceDataset(Path(args.data)/'val.jsonl',cfg),cfg,device,args.out)
