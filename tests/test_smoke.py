import json
import torch
from lagcd.model import LAGCD
from lagcd.data import ComplianceDataset
from lagcd.engine import collate
from lagcd.losses import objective

def test_forward_and_backward(tmp_path):
    cfg={'model':{'vocab_size':256,'d_model':32,'heads':4,'stat_layers':1,'har_layers':1,'gnn_layers':1,'ff_dim':64,'dropout':0.0,'max_nodes':4,'max_tokens_per_node':8,'max_context_tokens':8,'num_jurisdictions':2,'num_classes':4,'conflict_threshold':.3},
         'train':{'lambda_cls':1.,'lambda_reg':.5,'lambda_gen':.3,'lambda_calib':.1,'lambda_conflict':1.}}
    item={'provisions':['Article 1: comply','Article 2: exceptions'],'norms':[{'text':'shall do','jurisdiction_id':0},{'text':'shall not','jurisdiction_id':1}], 'lcot_answers':['a']*5,'label':2,'risk':65,'conflict_label':2,'rationale':'Article 1 applies.'}
    path=tmp_path/'a.jsonl';path.write_text(json.dumps(item)+'\n')
    b=collate([ComplianceDataset(path,cfg)[0]])
    model=LAGCD(cfg['model']);o=model(b)
    assert o['probabilities'].shape==(1,4)
    assert torch.allclose(o['probabilities'].sum(-1),torch.ones(1),atol=1e-5)
    loss,_=objective(o,b,model,cfg);loss.backward()
    assert any(p.grad is not None for p in model.parameters())
