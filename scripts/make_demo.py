"""Synthetic examples ONLY: never use for paper benchmark claims."""
import argparse,json,random
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='data/demo');p.add_argument('--n',type=int,default=120);args=p.parse_args()
random.seed(42); path=Path(args.out);path.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(args.n):
    label=i%4;j=i%2
    actor=['documented lawful procedures','minor process lapse','serious safeguard deficiency','prohibited unauthorized practice'][label]
    row={'id':f'synthetic_{i}','jurisdiction_id':j,'statutes':f'Article {10+label}. The controller shall document conditions and safeguards.',
         'provisions':[f'Article {10+label}: supervisory duties','Paragraph 1: record keeping and accountability'],
         'case_law':f'Case study on compliance {label}','policy':'Data protection and governance policy',
         'entities':['controller','data processor'],'scenario':f'AI deployment with {actor}; case {i}',
         'lcot_answers':[f'Step {s}: the case has {actor}.' for s in range(5)],
         'norms':[{'text':'Regulation A requires documentation','jurisdiction_id':0},{'text':'Regulation B forbids conflicting processing','jurisdiction_id':1}],
         'conflict_label':i%3,'label':label,'risk':[5.,30.,65.,95.][label],
         'rationale':f'Article {10+label} governs the listed facts.'}
    rows.append(row)
by_class={k:[] for k in range(4)}
for row in rows: by_class[row['label']].append(row)
parts={'train':[],'val':[],'test':[]}
for group in by_class.values():
    random.shuffle(group); a=max(1,int(.7*len(group))); b=min(len(group)-1,max(a+1,int(.8*len(group))))
    parts['train']+=group[:a];parts['val']+=group[a:b];parts['test']+=group[b:]
for part in parts.values(): random.shuffle(part)
for name,portion in parts.items():
    (path/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in portion),encoding='utf-8')
print('Wrote SYNTHETIC DEMONSTRATION data to',path)
