import csv
import os
import subprocess
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte
from sigver.preprocessing.normalize import preprocess_signature
import onnxruntime as ort

# pairs from README
GENUINE_FORGED = [
    ("002_09.PNG", "0108002_03.png"),
    ("001_10.PNG", "0201001_04.png"),
    ("004_11.PNG", "0105004_01.png"),
    ("004_15.PNG", "0105004_02.png"),
    ("004_21.PNG", "0124004_01.png"),
    ("001_11.PNG", "0201001_03.png"),
    ("003_06.PNG", "0126003_04.png"),
    ("003_02.PNG", "0121003_02.png"),
    ("002_16.PNG", "0110002_01.png"),
    ("002_23.PNG", "0118002_04.png"),
    ("003_20.PNG", "0121003_04.png"),
    ("002_11.PNG", "0108002_01.png"),
    ("001_08.PNG", "0119001_01.png"),
    ("001_11.PNG", "0119001_02.png"),
    ("003_08.PNG", "0206003_04.png"),
    ("001_22.PNG", "0119001_02.png"),
    ("004_19.PNG", "0124004_01.png"),
    ("001_11.PNG", "0201001_04.png"),
    ("004_05.PNG", "0103004_03.png"),
    ("004_17.PNG", "0103004_02.png"),
    ("003_02.PNG", "0121003_03.png"),
    ("003_23.PNG", "0121003_01.png"),
    ("004_14.PNG", "0124004_01.png"),
    ("001_13.PNG", "0119001_01.png"),
    ("001_10.PNG", "0201001_04.png"),
    ("003_07.PNG", "0121003_02.png"),
    ("001_12.PNG", "0119001_03.png"),
    ("004_24.PNG", "0105004_02.png"),
    ("001_10.PNG", "0201001_03.png"),
    ("003_02.PNG", "0121003_02.png"),
]

GENUINE_GENUINE = [
    ("002_01.PNG", "002_13.PNG"),
    ("001_19.PNG", "001_09.PNG"),
    ("002_04.PNG", "002_09.PNG"),
    ("003_17.PNG", "003_13.PNG"),
    ("003_03.PNG", "003_06.PNG"),
    ("003_18.PNG", "003_17.PNG"),
    ("002_12.PNG", "002_13.PNG"),
    ("002_18.PNG", "002_11.PNG"),
    ("002_02.PNG", "002_04.PNG"),
    ("003_01.PNG", "003_05.PNG"),
    ("002_13.PNG", "002_14.PNG"),
    ("002_06.PNG", "002_15.PNG"),
    ("004_15.PNG", "004_23.PNG"),
    ("002_06.PNG", "002_23.PNG"),
    ("001_15.PNG", "001_08.PNG"),
    ("002_23.PNG", "002_16.PNG"),
    ("001_21.PNG", "001_22.PNG"),
    ("004_21.PNG", "004_09.PNG"),
    ("002_18.PNG", "002_11.PNG"),
    ("002_07.PNG", "002_15.PNG"),
    ("003_16.PNG", "003_23.PNG"),
    ("003_23.PNG", "003_17.PNG"),
    ("002_04.PNG", "002_11.PNG"),
    ("004_05.PNG", "004_18.PNG"),
    ("003_05.PNG", "003_12.PNG"),
    ("004_21.PNG", "004_02.PNG"),
    ("003_15.PNG", "003_10.PNG"),
    ("004_11.PNG", "004_16.PNG"),
    ("003_14.PNG", "003_01.PNG"),
    ("001_10.PNG", "001_16.PNG"),
]

def load_features(session, image_path):
    img = img_as_ubyte(imread(image_path, as_gray=True))
    proc = preprocess_signature(img, (840, 1360))
    inp = proc.reshape(1, 1, 150, 220).astype(np.float32) / 255.0
    feat = session.run(None, {"input": inp})[0].astype(np.float64).ravel()
    return feat

session = ort.InferenceSession(os.path.join('models', 'signet.onnx'), providers=['CPUExecutionProvider'])

def distance(feat1, feat2):
    return float(np.linalg.norm(feat1 - feat2))

def dotnet_distance(ref_path, cand_path):
    cmd = ["dotnet", "FeatureDist/bin/Release/net9.0/FeatureDist.dll", ref_path, cand_path]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.join(os.getcwd(), "so")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return float(res.stdout.strip())

results = []
for pair_list, typ in [(GENUINE_FORGED, 'forged'), (GENUINE_GENUINE, 'genuine')]:
    for f1, f2 in pair_list:
        p1 = os.path.join('data', f1.split('_')[0], f1)
        if typ == 'forged':
            p2 = os.path.join('data', f1.split('_')[0] + '_forg', f2)
        else:
            p2 = os.path.join('data', f2.split('_')[0], f2)
        feat1 = load_features(session, p1)
        feat2 = load_features(session, p2)
        py_dist = distance(feat1, feat2)
        py_forg = py_dist > (0.8 if typ=='forged' else 6.0)
        dot_dist = dotnet_distance(p1, p2)
        dot_forg = dot_dist > (0.8 if typ=='forged' else 6.0)
        results.append([
            f1, f2,
            py_forg, f"{py_dist:.4f}",
            dot_forg, f"{dot_dist:.4f}",
            dot_forg != py_forg,
            f"{dot_dist - py_dist:.4f}"
        ])

with open('comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['File1', 'File2', 'IsForgeryPython', 'DistPython', 'IsForgeryDotnet', 'DistDotnet', 'Mismatch', 'DistDiff'])
    writer.writerows(results)
print('Saved comparison.csv with', len(results), 'rows')
