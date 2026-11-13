import torch
from torch import nn

class UncertaintyAwareCalibrator(nn.Module):
    def __init__(self,d,num_classes):
        super().__init__();self.cls=nn.Linear(3*d,num_classes);self.risk=nn.Linear(3*d,1)
        self.log_temperature=nn.Parameter(torch.zeros(()))
    def forward(self,z):
        logits=self.cls(z);t=self.log_temperature.exp().clamp(.05,20)
        return {'raw_logits':logits,'logits':logits/t,'probabilities':(logits/t).softmax(-1),
                'risk':self.risk(z).sigmoid().squeeze(-1)*100,'temperature':t}
