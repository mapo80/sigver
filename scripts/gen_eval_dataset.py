import os
import random
import csv
import time
import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_curve, auc, confusion_matrix, log_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODEL1 = os.path.join(ROOT, 'models', 'signet.onnx')
MODEL2 = os.path.join(ROOT, 'models', 'signet_f_lambda_0.95.onnx')

sess1 = ort.InferenceSession(MODEL1, providers=['CPUExecutionProvider'])
sess2 = ort.InferenceSession(MODEL2, providers=['CPUExecutionProvider'])

CACHE1 = {}
CACHE2 = {}

def preprocess(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = 255 - img
    h, w = img.shape
    canvas = np.full((840, 1360), 255, dtype=np.uint8)
    y0 = (840 - h) // 2
    x0 = (1360 - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img
    img = cv2.resize(canvas, (242, 170), interpolation=cv2.INTER_AREA)
    x1 = (242 - 220) // 2
    y1 = (170 - 150) // 2
    img = img[y1:y1+150, x1:x1+220]
    img = img.astype('float32') / 255.0
    return img

def features(sess, cache, path):
    if path not in cache:
        img = preprocess(path)
        feat = sess.run(None, {'input': img[None, None]})[0][0].astype('float32')
        feat /= np.linalg.norm(feat)
        cache[path] = feat
    return cache[path]

def distances(path1, path2):
    f1 = features(sess1, CACHE1, path1)
    f2 = features(sess1, CACHE1, path2)
    d1 = 1 - np.dot(f1, f2)
    g1 = features(sess2, CACHE2, path1)
    g2 = features(sess2, CACHE2, path2)
    d2 = 1 - np.dot(g1, g2)
    return d1, d2

def main():
    random.seed(123)
    users = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d)) and not d.endswith('_forg')]
    pairs = []
    # genuine vs forgery
    for _ in range(100):
        user = random.choice(users)
        genu_dir = os.path.join(DATA_DIR, user)
        forg_dir = os.path.join(DATA_DIR, user + '_forg')
        gfile = random.choice(os.listdir(genu_dir))
        ffile = random.choice(os.listdir(forg_dir))
        p1 = os.path.join(genu_dir, gfile)
        p2 = os.path.join(forg_dir, ffile)
        start = time.time()
        d1, d2 = distances(p1, p2)
        s_min = min(d1, d2)
        s_cal = s_min / 1.008
        decision = 0 if s_cal > 0.0010 else 1
        elapsed = (time.time() - start) * 1000
        pairs.append(['genuine-forgery', gfile, ffile, d1, d2, s_min, s_cal, decision, 0])
    # genuine vs genuine
    for _ in range(100):
        user = random.choice(users)
        genu_dir = os.path.join(DATA_DIR, user)
        files = os.listdir(genu_dir)
        a,b = random.sample(files, 2)
        p1 = os.path.join(genu_dir, a)
        p2 = os.path.join(genu_dir, b)
        start = time.time()
        d1, d2 = distances(p1, p2)
        s_min = min(d1, d2)
        s_cal = s_min / 1.008
        decision = 0 if s_cal > 0.0010 else 1
        elapsed = (time.time() - start) * 1000
        pairs.append(['genuine-genuine', a, b, d1, d2, s_min, s_cal, decision, 1])

    csv_path = os.path.join(ROOT, 'docs', 'eval_pairs.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type','file_a','file_b','d1','d2','s_min','s_cal','pred','label'])
        for row in pairs:
            writer.writerow(row)

    # compute metrics
    scores = np.array([row[6] for row in pairs])
    labels = np.array([row[8] for row in pairs])
    preds = np.array([row[7] for row in pairs])
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp) if tp+fp>0 else 0
    recall = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0

    fpr,tpr,thr = roc_curve(labels, -scores)
    roc_auc = auc(fpr,tpr)
    fnr = 1-tpr
    idx = np.argmin(np.abs(fpr-fnr))
    eer = (fpr[idx]+fnr[idx])/2
    eer_thr = thr[idx]

    prob = 1/(1+np.exp(scores))
    ll = log_loss(labels, prob)
    bins = np.linspace(0,1,11)
    inds = np.digitize(prob,bins)-1
    ece=0
    for i in range(10):
        m = inds==i
        if m.any():
            ece += m.mean()*abs(prob[m].mean()-labels[m].mean())

    metrics_csv = os.path.join(ROOT,'docs','eval_metrics.csv')
    with open(metrics_csv,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TP','FN','FP','TN','Accuracy','Precision','Recall','F1','AUC','EER','thrEER','LogLoss','ECE'])
        writer.writerow([tp,fn,fp,tn,accuracy,precision,recall,f1,roc_auc,eer,eer_thr,ll,ece])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr,tpr,label=f'AUC={roc_auc:.3f}')
    plt.xlabel('FPR');plt.ylabel('TPR');plt.legend(loc='lower right')
    plt.savefig(os.path.join(ROOT,'docs','eval_roc.png'))

    plt.figure()
    conf=[];acc=[]
    for i in range(10):
        m=inds==i
        if m.any():
            conf.append(prob[m].mean())
            acc.append(labels[m].mean())
        else:
            val=(bins[i]+bins[i+1])/2
            conf.append(val);acc.append(val)
    plt.plot(conf,acc,'o-');plt.plot([0,1],[0,1],'--')
    plt.xlabel('Confidence');plt.ylabel('Accuracy')
    plt.savefig(os.path.join(ROOT,'docs','eval_reliability.png'))

if __name__=='__main__':
    main()
