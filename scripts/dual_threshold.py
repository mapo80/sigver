import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

pairs = pd.read_csv('docs/eval_pairs.csv')
T = 1.008
s_cal = pairs[['d1','d2']].min(axis=1) / T
labels = pairs['label']

G = s_cal[labels==1]
F = s_cal[labels==0]

g_perc = 98
f_perc = 2
thr_low = np.percentile(G, g_perc)
thr_high = np.percentile(F, f_perc)
while thr_low >= thr_high and (g_perc > 50 or f_perc < 50):
    if g_perc > 50:
        g_perc -= 1
        thr_low = np.percentile(G, g_perc)
    if thr_low >= thr_high and f_perc < 50:
        f_perc += 1
        thr_high = np.percentile(F, f_perc)

pred = np.full(len(s_cal), -1)
pred[s_cal <= thr_low] = 1
pred[s_cal >= thr_high] = 0

rec_g = ((labels==1) & (pred==1)).sum() / (labels==1).sum()
rec_f = ((labels==0) & (pred==0)).sum() / (labels==0).sum()
zone_pct = (pred==-1).mean()

print('thr_low', thr_low, 'percentile', g_perc)
print('thr_high', thr_high, 'percentile', f_perc)
print('recall_genuine', rec_g)
print('recall_forgery', rec_f)
print('grey_zone_pct', zone_pct)

auto = pred!=-1
tp = ((labels==1)&(pred==1)&auto).sum()
fn = ((labels==1)&(pred==0)&auto).sum()
fp = ((labels==0)&(pred==1)&auto).sum()
tn = ((labels==0)&(pred==0)&auto).sum()

print('TP',tp,'FN',fn,'FP',fp,'TN',tn)

os.makedirs('docs', exist_ok=True)
with open('docs/dual_threshold_metrics.csv', 'w') as f:
    f.write('thr_low,thr_high,g_percentile,f_percentile,recall_genuine,recall_forgery,grey_pct,TP,FN,FP,TN\n')
    f.write(f'{thr_low},{thr_high},{g_perc},{f_perc},{rec_g},{rec_f},{zone_pct},{tp},{fn},{fp},{tn}\n')

# distribution plot
plt.figure()
plt.hist(G, bins=30, alpha=0.5, label='Genuine')
plt.hist(F, bins=30, alpha=0.5, label='Forgery')
plt.axvline(thr_low, color='green', linestyle='--', label='thr_low')
plt.axvline(thr_high, color='red', linestyle='--', label='thr_high')
plt.legend()
plt.xlabel('Calibrated score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('docs/dual_threshold_hist.png')
