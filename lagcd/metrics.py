import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,mean_absolute_error,confusion_matrix

def hard_ece(probs,y,n_bins=15):
    confidence=probs.max(1);correct=(probs.argmax(1)==y)
    bins=np.linspace(0,1,n_bins+1); result=0.
    for i in range(n_bins):
        m=(confidence>=bins[i]) & ((confidence<=bins[i+1]) if i==n_bins-1 else (confidence<bins[i+1]))
        if m.any(): result+=m.mean()*abs(confidence[m].mean()-correct[m].mean())
    return float(result)

def metrics(y,probs,risk_pred=None,risk=None,risk_available=None):
    y=np.asarray(y);p=np.asarray(probs);pred=p.argmax(1)
    result={'macro_f1':float(f1_score(y,pred,labels=list(range(p.shape[1])),average='macro',zero_division=0)),
            'per_class_f1':[float(x) for x in f1_score(y,pred,labels=list(range(p.shape[1])),average=None,zero_division=0)],
            'ece':hard_ece(p,y),'confusion_matrix':confusion_matrix(y,pred,labels=list(range(p.shape[1]))).tolist()}
    if len(np.unique(y))!=p.shape[1]: result['macro_auroc']=None
    else:
        try:
            auc=float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=list(range(p.shape[1]))))
            result['macro_auroc']=auc if np.isfinite(auc) else None
        except ValueError: result['macro_auroc']=None
    if risk_pred is not None and risk_available is not None:
        keep=np.asarray(risk_available,dtype=bool)
        result['risk_mae']=float(mean_absolute_error(np.asarray(risk)[keep],np.asarray(risk_pred)[keep])) if keep.any() else None
    return result
