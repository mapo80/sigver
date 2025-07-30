import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, log_loss
import sys, os

pairs = pd.read_csv('docs/eval_pairs.csv')
score = pairs['s_cal']
inv_score = -score  # higher values indicate genuine
label = pairs['label']

# compute metrics at fixed threshold 0.0056
def metrics_at(thr):
    pred = (score <= thr).astype(int)
    tp = ((pred == 1) & (label == 1)).sum()
    fn = ((pred == 0) & (label == 1)).sum()
    fp = ((pred == 1) & (label == 0)).sum()
    tn = ((pred == 0) & (label == 0)).sum()
    acc = (tp + tn) / len(label)
    prec = tp / (tp + fp) if tp+fp>0 else 0
    rec = tp / (tp + fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    fpr = fp/(fp+tn) if fp+tn>0 else 0
    fnr = fn/(fn+tp) if fn+tp>0 else 0
    return tp, fn, fp, tn, acc, prec, rec, f1, fpr, fnr

tp,fn,fp,tn,acc,prec,rec,f1,fpr,fnr = metrics_at(0.0056)
roc_fpr, roc_tpr, thr = roc_curve(label, inv_score, pos_label=1)
roc_auc = auc(roc_fpr, roc_tpr)
fnr_curve = 1 - roc_tpr
eer_idx = np.argmin(np.abs(roc_fpr - fnr_curve))
eer = (roc_fpr[eer_idx] + fnr_curve[eer_idx])/2
thr_eer = thr[eer_idx]
prob = 1/(1+np.exp(inv_score))
ll = log_loss(label, prob)
bins = np.linspace(0,1,11)
inds = np.digitize(prob,bins)-1
ece=0
for i in range(10):
    m = inds==i
    if m.any():
        ece += m.mean()*abs(prob[m].mean()-label[m].mean())

# sweep threshold
s_min = score.min()
s_max = score.max()
ths = np.linspace(s_min,s_max,200)
results=[]
for t in ths:
    tp1,fn1,fp1,tn1,acc1,prec1,rec1,f11,fpr1,fnr1 = metrics_at(t)
    results.append((t,fpr1,rec1,prec1,f11))

results = pd.DataFrame(results, columns=['thr','fpr','recall','precision','f1'])
best = results[results['fpr']<=0.01]
if not best.empty:
    best_thr = best.sort_values('recall',ascending=False).iloc[0]['thr']
    tp2,fn2,fp2,tn2,acc2,prec2,rec2,f12,fpr2,fnr2 = metrics_at(best_thr)
else:
    best_thr = results.sort_values('f1',ascending=False).iloc[0]['thr']
    tp2,fn2,fp2,tn2,acc2,prec2,rec2,f12,fpr2,fnr2 = metrics_at(best_thr)

# save csv
pd.DataFrame({
    'TP':[tp], 'FN':[fn], 'FP':[fp], 'TN':[tn], 'Accuracy':[acc], 'Precision':[prec], 'Recall':[rec], 'F1':[f1],
    'AUC':[roc_auc], 'EER':[eer], 'thrEER':[thr_eer], 'LogLoss':[ll], 'ECE':[ece]
}).to_csv('docs/eval_metrics.csv', index=False)

# store sweep results
results.to_csv('docs/threshold_sweep.csv',index=False)

plt.figure()
plt.plot(roc_fpr, roc_tpr, label=f'AUC={roc_auc:.3f}')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(loc='lower right')
plt.savefig('docs/eval_roc.png')

plt.figure()
conf=[]; accs=[]
for i in range(10):
    m = inds==i
    if m.any():
        conf.append(prob[m].mean()); accs.append(label[m].mean())
    else:
        val=(bins[i]+bins[i+1])/2; conf.append(val); accs.append(val)
plt.plot(conf,accs,'o-'); plt.plot([0,1],[0,1],'--')
plt.xlabel('Confidence'); plt.ylabel('Accuracy')
plt.savefig('docs/eval_reliability.png')

print('best_thr',best_thr)
print('best_precision',prec2,'recall',rec2,'fpr',fpr2)
