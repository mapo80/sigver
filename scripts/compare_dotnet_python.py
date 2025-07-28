import os
import random
import subprocess
import json
from skimage.io import imread
from skimage import img_as_ubyte
import torch
import numpy as np
from sigver.preprocessing.normalize import preprocess_signature
from sigver.featurelearning.models import SigNet

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PY = os.path.join(os.path.dirname(__file__), '..', 'models', 'signet.pth')
MODEL_NET = os.path.join(os.path.dirname(__file__), '..', 'models', 'signet.onnx')
CLI_DLL = os.path.join(os.path.dirname(__file__), '..', 'SigCompare', 'bin', 'Release', 'net9.0', 'SigCompare.dll')

# pair generation matching SignatureTests.cs

def genuine_forged_pairs(seed=42, n=30):
    rng = random.Random(seed)
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and not f.endswith('_forg')]
    for _ in range(n):
        folder = rng.choice(folders)
        gen_dir = os.path.join(DATA_DIR, folder)
        forged_dir = os.path.join(DATA_DIR, folder + '_forg')
        gen = rng.choice(os.listdir(gen_dir))
        forg = rng.choice(os.listdir(forged_dir))
        yield os.path.join(gen_dir, gen), os.path.join(forged_dir, forg), 0.35, True


def genuine_pairs(seed=99, n=30):
    rng = random.Random(seed)
    folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and not f.endswith('_forg')]
    for _ in range(n):
        folder = rng.choice(folders)
        gen_dir = os.path.join(DATA_DIR, folder)
        files = os.listdir(gen_dir)
        if len(files) < 2:
            continue
        first = rng.choice(files)
        second = rng.choice(files)
        while second == first:
            second = rng.choice(files)
        yield os.path.join(gen_dir, first), os.path.join(gen_dir, second), 0.35, False


def load_model():
    device = torch.device('cpu')
    state_dict, _, _ = torch.load(MODEL_PY, map_location=device)
    model = SigNet().to(device).eval()
    model.load_state_dict(state_dict)
    return model, device


def extract_features(model, device, path):
    img = img_as_ubyte(imread(path, as_gray=True))
    proc = preprocess_signature(img, (840, 1360))
    inp = torch.from_numpy(proc).view(1, 1, 150, 220).float().div(255).to(device)
    with torch.no_grad():
        feats = model(inp).cpu().numpy().flatten()
    return feats


def distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(1 - np.dot(a, b))


def dotnet_result(img1, img2, thr):
    cmd = ['dotnet', CLI_DLL, MODEL_NET, img1, img2, str(thr)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    dist_str, is_forg = res.stdout.strip().split()
    return float(dist_str), is_forg == 'True'


def main():
    model, device = load_model()
    results = []
    for img1, img2, thr, expected in list(genuine_forged_pairs()) + list(genuine_pairs()):
        f1 = extract_features(model, device, img1)
        f2 = extract_features(model, device, img2)
        dist_py = distance(f1, f2)
        is_forg_py = dist_py > thr
        dist_net, is_forg_net = dotnet_result(img1, img2, thr)
        results.append({
            'img1': os.path.basename(img1),
            'img2': os.path.basename(img2),
            'expected': expected,
            'is_forg_py': is_forg_py,
            'match_py': is_forg_py == expected,
            'dist_py': dist_py,
            'is_forg_net': is_forg_net,
            'match_net': is_forg_net == expected,
            'dist_net': dist_net,
            'diff': dist_net - dist_py
        })
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
