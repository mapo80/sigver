import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, log_loss
import os

# load pairs
pairs = pd.read_csv('docs/eval_pairs.csv')
# compute calibrated score
T = 1.008
s_cal = pairs[['d1','d2']].min(axis=1) / T
labels = pairs['label']

thr_min = s_cal.min()
thr_max = s_cal.max()
thresholds = np.linspace(thr_min, thr_max, 300)
results = []

def compute_metrics(thr):
    pred = (s_cal <= thr).astype(int)
    tp = ((pred==1) & (labels==1)).sum()
    fn = ((pred==0) & (labels==1)).sum()
    fp = ((pred==1) & (labels==0)).sum()
    tn = ((pred==0) & (labels==0)).sum()
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    fpr = fp/(fp+tn) if fp+tn>0 else 0
    fnr = fn/(fn+tp) if fn+tp>0 else 0
    return tp, fn, fp, tn, prec, rec, f1, fpr, fnr

for thr in thresholds:
    metrics = compute_metrics(thr)
    results.append((thr,) + metrics)

cols = ['threshold','TP','FN','FP','TN','Precision','Recall','F1','FPR','FNR']
res_df = pd.DataFrame(results, columns=cols)
res_df.to_csv('docs/threshold_sweep_results.csv', index=False)

# selection according to criteria
candidates = res_df[res_df['FPR'] <= 0.01]
if not candidates.empty:
    best = candidates.sort_values(['Recall','F1'], ascending=False).iloc[0]
else:
    # if no threshold meets the 1% FPR requirement, fall back to the one with best F1
    best = res_df.sort_values('F1', ascending=False).iloc[0]
opt_thr = best['threshold']

# final metrics
tp, fn, fp, tn, prec, rec, f1, fpr, fnr = compute_metrics(opt_thr)
accuracy = (tp+tn)/(tp+tn+fp+fn)

fpr_curve, tpr_curve, thr_curve = roc_curve(labels, -s_cal)
roc_auc = auc(fpr_curve, tpr_curve)
fnr_curve = 1 - tpr_curve
eer_idx = np.argmin(np.abs(fpr_curve - fnr_curve))
eer = (fpr_curve[eer_idx] + fnr_curve[eer_idx])/2

prob = 1/(1+np.exp(s_cal))
ll = log_loss(labels, prob)
bins = np.linspace(0,1,11)
inds = np.digitize(prob,bins)-1
ece = 0
cal_x=[]; cal_y=[]
for i in range(10):
    m = inds==i
    if m.any():
        conf = prob[m].mean(); acc = labels[m].mean();
        ece += m.mean()*abs(acc-conf)
        cal_x.append(conf); cal_y.append(acc)
    else:
        val = (bins[i]+bins[i+1])/2
        cal_x.append(val); cal_y.append(val)

pd.DataFrame({
    'Threshold':[opt_thr],
    'TP':[tp],'FN':[fn],'FP':[fp],'TN':[tn],
    'Accuracy':[accuracy],'Precision':[prec],'Recall':[rec],'F1':[f1],
    'FPR':[fpr],'FNR':[fnr],'AUC':[roc_auc],'EER':[eer],
    'LogLoss':[ll],'ECE':[ece]
}).to_csv('docs/eval_metrics.csv', index=False)

plt.figure()
plt.plot(fpr_curve, tpr_curve, label=f'AUC={roc_auc:.3f}')
plt.scatter(fpr, rec, color='red')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(loc='lower right')
plt.savefig('docs/eval_roc.png')

plt.figure()
plt.plot(cal_x, cal_y, 'o-')
plt.plot([0,1],[0,1],'--')
plt.xlabel('Confidence'); plt.ylabel('Accuracy')
plt.savefig('docs/eval_reliability.png')

print('opt_thr', opt_thr, 'precision', prec, 'recall', rec, 'fpr', fpr)
