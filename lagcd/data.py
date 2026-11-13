"""Expected JSONL schema documented in README. Deterministic, offline byte hashing."""
import hashlib
import json
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset

STREAMS = ('stat', 'case', 'policy', 'entity')
LABELS = ['compliant', 'minor_risk', 'major_risk', 'non_compliant']

def tokens(s):
    return re.findall(r"[\w]+|[^\w\s]", str(s).lower(), flags=re.UNICODE)

def hash_token(word, vocab_size):
    return 2 + (int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), 'big') % (vocab_size - 2))

def encode_text(text, max_len, vocab_size):
    ids = [hash_token(t, vocab_size) for t in tokens(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))

def provision_split(text):
    parts = re.split(r'(?=\b(?:Article|Section|Paragraph)\s+\d+\b)', str(text), flags=re.I)
    return [p.strip() for p in parts if p.strip()] or [str(text)]

class ComplianceDataset(Dataset):
    def __init__(self, path, cfg):
        self.items = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
        self.cfg = cfg
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        r = self.items[idx]; m = self.cfg['model']; v = m['vocab_size']; n = m['max_nodes']; t = m['max_tokens_per_node']
        provisions = r.get('provisions') or provision_split(r.get('statutes', ''))
        provisions = provisions[:n]
        node_ids = [encode_text(p, t, v) for p in provisions]
        node_ids += [[0]*t for _ in range(n-len(node_ids))]
        node_mask = [1]*len(provisions)+[0]*(n-len(provisions))
        # Default chain for article/paragraph hierarchy; explicit edges can override.
        edges = r.get('edges', [[i-1,i] for i in range(1,len(provisions))])
        adj = torch.eye(n)
        for a,b in edges:
            if 0<=a<len(provisions) and 0<=b<len(provisions): adj[a,b]=1;adj[b,a]=1
        ctx = []
        for key in ('case_law', 'policy', 'entities'):
            value = r.get(key, '')
            if isinstance(value,list): value = ' '.join(map(str,value))
            ctx.append(encode_text(value, m['max_context_tokens'], v))
        questions = r.get('lcot_answers', ['']*5)
        if len(questions)!=5: raise ValueError('lcot_answers must have exactly 5 items')
        reasoning = [encode_text(a, m['max_context_tokens'], v) for a in questions]
        norms = r.get('norms', [])[:n]
        norm_ids=[encode_text(x.get('text',''),t,v) for x in norms]+[[0]*t for _ in range(n-len(norms))]
        norm_j=[int(x.get('jurisdiction_id',0)) % m['num_jurisdictions'] for x in norms]+[0]*(n-len(norms))
        norm_mask=[1]*len(norms)+[0]*(n-len(norms))
        label = r.get('label', 0)
        if isinstance(label,str): label=LABELS.index(label)
        return {'nodes':torch.tensor(node_ids),'node_mask':torch.tensor(node_mask,dtype=torch.bool),
                'adj':adj,'context':torch.tensor(ctx),'answers':torch.tensor(reasoning),
                'norms':torch.tensor(norm_ids),'norm_jurisdictions':torch.tensor(norm_j),
                'norm_mask':torch.tensor(norm_mask,dtype=torch.bool),
                'jurisdiction':torch.tensor(int(r.get('jurisdiction_id',0)) % m['num_jurisdictions']),
                'label':torch.tensor(int(label)), 'risk':torch.tensor(float(r.get('risk',0.))/100.0),
                'has_risk':torch.tensor('risk' in r),
                'conflict_label':torch.tensor(int(r.get('conflict_label',-1))),
                'rationale':str(r.get('rationale','')), 'id':str(r.get('id',idx))}
