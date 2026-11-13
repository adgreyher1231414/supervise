"""Optional LCoT and rationale generation. Mock mode is for software smoke tests only."""
from .lcot import QUESTIONS

class LegalReasoner:
    def __init__(self, provider='mock', model='claude-sonnet-4-6', max_tokens=256):
        self.provider=provider; self.model=model; self.max_tokens=max_tokens
        if provider=='anthropic':
            try: import anthropic
            except ImportError as e: raise RuntimeError('pip install anthropic') from e
            self.client=anthropic.Anthropic()
        elif provider!='mock': raise ValueError('provider must be mock or anthropic')
    def _complete(self,prompt):
        if self.provider=='mock': return 'Unverified mock answer; review the cited legal provisions manually.'
        response=self.client.messages.create(model=self.model,max_tokens=self.max_tokens,messages=[{'role':'user','content':prompt}])
        return '\n'.join(block.text for block in response.content if getattr(block,'type',None)=='text')
    def steps(self,scenario,statutes):
        answers=[]
        for question in QUESTIONS:
            prompt=('You are a legal research assistant. Use only the provided provisions. '
                    'If evidence is missing say so, never invent article numbers. '
                    'This is not legal advice.\nScenario: '+scenario+'\nProvisions: '+statutes+'\nQuestion: '+question)
            answers.append(self._complete(prompt))
        return answers
    def rationale(self,scenario,statutes,verdict,answers):
        prompt=(f'Draft a concise, evidence-grounded compliance analysis, not legal advice. '
                f'Quote only article identifiers present in the supplied text.\nScenario: {scenario}\n'
                f'Statutes: {statutes}\nPredicted class: {verdict}\nAssessment notes: {answers}')
        return self._complete(prompt)
