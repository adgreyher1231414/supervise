from torch import nn

QUESTIONS = [
 'What law applies to this scenario?',
 'What obligations are imposed by the cited provisions?',
 'Are relevant statutory conditions satisfied?',
 'Do any exemptions or exceptions apply?',
 'What compliance verdict is supported by the cited evidence?'
]
class LegalChainOfThought(nn.Module):
    def __init__(self,d):
        super().__init__();self.answer_proj=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d)); self.gru=nn.GRU(d,d,batch_first=True)
    def forward(self,answer_embeddings):
        states=self.answer_proj(answer_embeddings)
        _,h=self.gru(states)
        return h[-1], states
