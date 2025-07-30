import os
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_base = os.path.join(root, 'models', 'signet.onnx')
model_fine = os.path.join(root, 'models', 'signet_f_lambda_0.95.onnx')
data_dir = os.path.join(root, 'data')

sess1 = ort.InferenceSession(model_base, providers=['CPUExecutionProvider'])
sess2 = ort.InferenceSession(model_fine, providers=['CPUExecutionProvider'])

cache1, cache2 = {}, {}


def preprocess(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = 255 - img
    h, w = img.shape
    canvas = np.full((840, 1360), 255, dtype=np.uint8)
    y0 = (840 - h) // 2
    x0 = (1360 - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    img = cv2.resize(canvas, (242, 170), interpolation=cv2.INTER_AREA)
    x1 = (242 - 220) // 2
    y1 = (170 - 150) // 2
    img = img[y1:y1 + 150, x1:x1 + 220]
    return img.astype('float32') / 255.0


def extract(sess, path, cache):
    if path in cache:
        return cache[path]
    img = preprocess(path)
    inp = img[None, None]
    feat = sess.run(None, {'input': inp})[0][0].astype('float32')
    feat /= np.linalg.norm(feat)
    cache[path] = feat
    return feat


pairs_gen, pairs_for = [], []
for d in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, d)) or d.endswith('_forg'):
        continue
    genu_dir = os.path.join(data_dir, d)
    forg_dir = os.path.join(data_dir, d + '_forg')
    genu_files = [os.path.join(genu_dir, f) for f in os.listdir(genu_dir)]
    forg_files = []
    if os.path.isdir(forg_dir):
        forg_files = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir)]
    for i in range(len(genu_files)):
        for j in range(i + 1, len(genu_files)):
            pairs_gen.append((genu_files[i], genu_files[j]))
    for g in genu_files:
        for f in forg_files:
            pairs_for.append((g, f))


def distance(pair):
    a, b = pair
    d1 = 1 - np.dot(extract(sess1, a, cache1), extract(sess1, b, cache1))
    d2 = 1 - np.dot(extract(sess2, a, cache2), extract(sess2, b, cache2))
    return d1, d2


D1_gen, D2_gen, D1_for, D2_for = [], [], [], []
for p in pairs_gen:
    d1, d2 = distance(p)
    D1_gen.append(d1)
    D2_gen.append(d2)
for p in pairs_for:
    d1, d2 = distance(p)
    D1_for.append(d1)
    D2_for.append(d2)

D1_gen = np.array(D1_gen)
D2_gen = np.array(D2_gen)
D1_for = np.array(D1_for)
D2_for = np.array(D2_for)
Smin_gen = np.minimum(D1_gen, D2_gen)
Smin_for = np.minimum(D1_for, D2_for)

scores = np.concatenate([Smin_gen, Smin_for])
labels = np.concatenate([np.ones_like(Smin_gen), np.zeros_like(Smin_for)])

T = 1.0
for _ in range(200):
    grad = 0.0
    for s, l in zip(scores, labels):
        p = 1 / (1 + np.exp(s / T))
        grad += (p - l) * (-s) / (T ** 2)
    grad /= len(scores)
    T -= 0.01 * grad
    if T < 0.1:
        T = 0.1

probs = 1 / (1 + np.exp(scores / T))
logloss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

# optional logistic regression (a*s + b)
a = 1.0
b = 0.0
for _ in range(200):
    grad_a = 0.0
    grad_b = 0.0
    for s,l in zip(scores,labels):
        p = 1/(1+np.exp(-(a*s+b)))
        e = p - l
        grad_a += e*s
        grad_b += e
    grad_a/=len(scores)
    grad_b/=len(scores)
    a -= 0.01*grad_a
    b -= 0.01*grad_b

probs_lr = 1/(1+np.exp(-(a*scores+b)))
logloss_lr = -np.mean(labels*np.log(probs_lr)+(1-labels)*np.log(1-probs_lr))
bins_lr = np.linspace(0,1,11)
inds_lr = np.digitize(probs_lr,bins_lr)-1
ece_lr=0.0
for i in range(10):
    m=inds_lr==i
    if m.any():
        ece_lr += m.mean()*abs(probs_lr[m].mean()-labels[m].mean())

bins = np.linspace(0, 1, 11)
inds = np.digitize(probs, bins) - 1
ece = 0.0
for i in range(10):
    mask = inds == i
    if mask.any():
        ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())

fpr, tpr, _ = roc_curve(labels, -scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.savefig(os.path.join(root, 'docs', 'roc.png'))

plt.figure()
conf, acc = [], []
for i in range(10):
    mask = inds == i
    if mask.any():
        conf.append(probs[mask].mean())
        acc.append(labels[mask].mean())
    else:
        val = (bins[i] + bins[i + 1]) / 2
        conf.append(val)
        acc.append(val)
plt.plot(conf, acc, 'o-')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(root, 'docs', 'reliability.png'))

N = len(Smin_for)
for_sorted = np.sort(Smin_for)
thr1 = for_sorted[int(0.01 * N)]
thr5 = for_sorted[int(0.05 * N)]
recall1 = (Smin_gen <= thr1).mean()
prec1 = (Smin_gen <= thr1).sum() / ((Smin_gen <= thr1).sum() + (Smin_for <= thr1).sum())
recall5 = (Smin_gen <= thr5).mean()
prec5 = (Smin_gen <= thr5).sum() / ((Smin_gen <= thr5).sum() + (Smin_for <= thr5).sum())

print('Temperature', T)
print('LogLoss', logloss)
print('ECE', ece)
print('thr1', thr1, 'prec1', prec1, 'recall1', recall1)
print('thr5', thr5, 'prec5', prec5, 'recall5', recall5)
print('LogLoss_LR', logloss_lr)
print('ECE_LR', ece_lr)

np.savez(os.path.join(root, 'docs', 'metrics.npz'),
         T=T, logloss=logloss, ece=ece, thr1=thr1, thr5=thr5,
         prec1=prec1, prec5=prec5, recall1=recall1, recall5=recall5,
         auc=roc_auc, logloss_lr=logloss_lr, ece_lr=ece_lr, a=a, b=b)
