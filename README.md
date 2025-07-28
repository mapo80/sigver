# SigVer: Offline Signature Verification

SigVer provides neural models and training scripts for verifying handwritten signatures on paper. The original project was created to learn writer–independent representations and classifiers in PyTorch. This repository also demonstrates how to export the trained weights to ONNX so the models can be used in other runtimes such as .NET.

## Quick installation

1. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Export the pretrained weights to ONNX:
   ```bash
   python convert_to_onnx.py
   ```
   The script produces `models/signet.onnx` and `models/signet_f_lambda_0.95.onnx`.
3. Install the .NET SDK (required to build `SigVerSdk`) if it is not already
   available:
   ```bash
   ./dotnet-install.sh --channel 8.0
   ```

## Using PyTorch models

To extract features from a signature using PyTorch:

```python
import torch
from sigver.featurelearning.models import SigNet

state_dict, _, _ = torch.load('models/signet.pth')
model = SigNet().eval()
model.load_state_dict(state_dict)
with torch.no_grad():
    feats = model(image_tensor)
```

## ONNX models

The exported ONNX networks are equivalent to the PyTorch weights `signet.pth` and `signet_f_lambda_0.95.pth`. They output a 2048‑dimensional feature vector and can be used with any ONNX runtime.

## .NET SDK

The `SigVerSdk` folder contains a small .NET library that relies on `onnxruntime`. It loads an ONNX model and computes signature features. Example usage in C#:

```csharp
using var verifier = new SigVerSdk.SigVerifier("models/signet.onnx");
float[] features = verifier.ExtractFeatures("data/a1.png");
```

`ExtractFeatures` now mirrors the Python preprocessing pipeline. Images are
centered on a 1360×840 canvas, cleaned with Otsu thresholding, resized to
170×242 and then cropped to 150×220 before the model is invoked. The method also
validates the output, throwing an exception when the model returns NaN or
infinite values.

The unit tests in `SigVerSdk.Tests` illustrate a simple verification scenario comparing two signatures.

If the required SDK is missing, run the provided install script before building:

```bash
./dotnet-install.sh --channel 8.0
```

## Data preprocessing

Training scripts expect data in a single `.npz` file containing:

* `x`: signature images (N × 1 × H × W)
* `y`: the user that produced each signature (N)
* `yforg`: whether the image is a forgery (1) or genuine (0)

Common datasets can be processed with `sigver.datasets.process_dataset`. For example, the MCYT dataset can be prepared as follows:

```bash
python -m sigver.preprocessing.process_dataset --dataset mcyt \
  --path MCYT-ORIGINAL/MCYToffline75original --save-path mcyt_170_242.npz
```
During training a random 150×220 crop is used; at test time the center crop is applied.

## Training writer‑independent networks

Two loss functions are implemented as described in the original paper:

* **SigNet** – uses only genuine signatures.
* **SigNet‑F** – incorporates knowledge of forgeries (`--forg`).

Example commands:

```bash
python -m sigver.featurelearning.train --model signet --dataset-path <data.npz> \
  --users 300 881 --epochs 60 --logdir signet

python -m sigver.featurelearning.train --model signet --dataset-path <data.npz> \
  --users 300 881 --epochs 60 --forg --lamb 0.95 --logdir signet_f_lamb0.95
```

All command line options are listed with `python -m sigver.featurelearning.train --help`. Real‑time monitoring is available with the `--visdom-port` option.

## Training writer‑dependent classifiers

To train and evaluate WD classifiers:

```bash
python -m sigver.wd.test -m signet --model-path <path/to/trained_model> \
    --data-path <path/to/data> --save-path <path/to/save> \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12
```

The script performs K random splits (default 10) and stores a pickle file with results containing metrics such as FAR and FRR as well as the predictions for each image.

## Pre‑trained models

Pre‑trained weights are available for convenience:
* [SigNet](https://drive.google.com/open?id=1l8NFdxSvQSLb2QTv71E6bKcTgvShKPpx)
* [SigNet‑F lambda 0.95](https://drive.google.com/open?id=1ifaUiPtP1muMjt8Tkrv7yJj7we8ttncW)

These weights were trained with pixel values in the range [0, 1]. Divide each pixel by 255 before inference, e.g. `x = x.float().div(255)`. `torchvision.transforms.ToTensor` already performs this division.

## Example dataset

The repository contains a small dataset in the `data` directory. Each folder such
as `001` holds genuine signatures from a single user. The matching folder
`001_forg` stores forged signatures produced by other writers. Every user has 24
genuine samples and roughly 8–12 forgeries. Below is a preview showing ten pairs
from the `001` and `002` folders—genuine signatures on the left and forgeries on
the right.

<table>
  <tr>
    <td><img src="data/001/001_01.PNG" width="150"></td>
    <td><img src="data/001_forg/0119001_01.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/001/001_02.PNG" width="150"></td>
    <td><img src="data/001_forg/0119001_02.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/001/001_03.PNG" width="150"></td>
    <td><img src="data/001_forg/0119001_03.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/001/001_04.PNG" width="150"></td>
    <td><img src="data/001_forg/0119001_04.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_01.PNG" width="150"></td>
    <td><img src="data/002_forg/0108002_01.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_02.PNG" width="150"></td>
    <td><img src="data/002_forg/0108002_02.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_03.PNG" width="150"></td>
    <td><img src="data/002_forg/0108002_03.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_04.PNG" width="150"></td>
    <td><img src="data/002_forg/0108002_04.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_05.PNG" width="150"></td>
    <td><img src="data/002_forg/0110002_01.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="data/002/002_06.PNG" width="150"></td>
    <td><img src="data/002_forg/0110002_02.png" width="150"></td>
  </tr>
</table>

### Preprocessing comparison

The verifier preprocesses every input image before running the ONNX model. The
Python library and the C# implementation now follow the same steps: centering
the signature on an 840×1360 canvas, removing the background with Otsu
thresholding, resizing to 170×242 and cropping a 150×220 window. The tables
below show the resulting images for those involved in the failing test as well
as ten examples from passing tests.

#### Failing test images

| Original | Python | .NET |
|---------|--------|------|
| `001_01.PNG` | ![](docs/preprocess/python/001_01_py.png) | ![](docs/preprocess/dotnet/001_01_dotnet.png) |
| `001_02.PNG` | ![](docs/preprocess/python/001_02_py.png) | ![](docs/preprocess/dotnet/001_02_dotnet.png) |
| `002_01.PNG` | ![](docs/preprocess/python/002_01_py.png) | ![](docs/preprocess/dotnet/002_01_dotnet.png) |
| `002_02.PNG` | ![](docs/preprocess/python/002_02_py.png) | ![](docs/preprocess/dotnet/002_02_dotnet.png) |

#### Passing examples

| Original | Python | .NET |
|---------|--------|------|
| `004_04.PNG` | ![](docs/preprocess/python/004_04_py.png) | ![](docs/preprocess/dotnet/004_04_dotnet.png) |
| `0105004_03.png` | ![](docs/preprocess/python/0105004_03_py.png) | ![](docs/preprocess/dotnet/0105004_03_dotnet.png) |
| `003_01.PNG` | ![](docs/preprocess/python/003_01_py.png) | ![](docs/preprocess/dotnet/003_01_dotnet.png) |
| `0126003_02.png` | ![](docs/preprocess/python/0126003_02_py.png) | ![](docs/preprocess/dotnet/0126003_02_dotnet.png) |
| `004_23.PNG` | ![](docs/preprocess/python/004_23_py.png) | ![](docs/preprocess/dotnet/004_23_dotnet.png) |
| `0103004_04.png` | ![](docs/preprocess/python/0103004_04_py.png) | ![](docs/preprocess/dotnet/0103004_04_dotnet.png) |
| `001_06.PNG` | ![](docs/preprocess/python/001_06_py.png) | ![](docs/preprocess/dotnet/001_06_dotnet.png) |
| `0119001_02.png` | ![](docs/preprocess/python/0119001_02_py.png) | ![](docs/preprocess/dotnet/0119001_02_dotnet.png) |
| `002_19.PNG` | ![](docs/preprocess/python/002_19_py.png) | ![](docs/preprocess/dotnet/002_19_dotnet.png) |
| `0118002_02.png` | ![](docs/preprocess/python/0118002_02_py.png) | ![](docs/preprocess/dotnet/0118002_02_dotnet.png) |

### Automated C# tests

The SDK includes a test suite that randomly generates signature pairs from the
`data` directory. Each user has a folder such as `001` with genuine signatures
and a matching folder `001_forg` with forgeries. Using xUnit, thirty pairs are
sampled in two categories:

1. **Genuine vs forged** — one genuine signature from `XXX` against one forged
   signature from `XXX_forg`.
2. **Genuine vs genuine** — two different genuine signatures from the same
   folder `XXX`.

Random selection uses a fixed seed so results are reproducible. The tests assert
that forged signatures are detected and that genuine pairs match. Running
`dotnet test` executes all sixty comparisons and logs the verification time for
each pair.

## Test results

The tables below show the outcome of running the signature verifier on 60
randomly selected pairs from the `data` directory. Half of the pairs compare a
genuine signature with a forged one, while the other half compare two genuine
signatures of the same user. Each comparison reports whether the verifier
behaved as expected and how long it took.

### Genuine vs forged

| Genuine | Forged | Detected | Time (ms) |
|---------|--------|----------|-----------|
| 002_09.PNG | 0108002_03.png | True | 18 |
| 001_10.PNG | 0201001_04.png | True | 22 |
| 004_11.PNG | 0105004_01.png | True | 14 |
| 004_15.PNG | 0105004_02.png | True | 14 |
| 004_21.PNG | 0124004_01.png | True | 14 |
| 001_11.PNG | 0201001_03.png | True | 28 |
| 003_06.PNG | 0126003_04.png | True | 23 |
| 003_02.PNG | 0121003_02.png | True | 32 |
| 002_16.PNG | 0110002_01.png | True | 16 |
| 002_23.PNG | 0118002_04.png | True | 17 |
| 003_20.PNG | 0121003_04.png | True | 20 |
| 002_11.PNG | 0108002_01.png | True | 26 |
| 001_08.PNG | 0119001_01.png | True | 17 |
| 001_11.PNG | 0119001_02.png | True | 20 |
| 003_08.PNG | 0206003_04.png | True | 16 |
| 001_22.PNG | 0119001_02.png | True | 18 |
| 004_19.PNG | 0124004_01.png | True | 15 |
| 001_11.PNG | 0201001_04.png | True | 23 |
| 004_05.PNG | 0103004_03.png | True | 16 |
| 004_17.PNG | 0103004_02.png | True | 16 |
| 003_02.PNG | 0121003_03.png | True | 16 |
| 003_23.PNG | 0121003_01.png | True | 16 |
| 004_14.PNG | 0124004_01.png | True | 19 |
| 001_13.PNG | 0119001_01.png | True | 22 |
| 001_10.PNG | 0201001_04.png | True | 20 |
| 003_07.PNG | 0121003_02.png | True | 17 |
| 001_12.PNG | 0119001_03.png | True | 20 |
| 004_24.PNG | 0105004_02.png | True | 15 |
| 001_10.PNG | 0201001_03.png | True | 18 |
| 003_02.PNG | 0121003_02.png | True | 16 |

### Genuine vs genuine

| Reference | Candidate | Match | Time (ms) |
|-----------|-----------|-------|-----------|
| 002_01.PNG | 002_13.PNG | True | 95 |
| 001_19.PNG | 001_09.PNG | True | 33 |
| 002_04.PNG | 002_09.PNG | True | 17 |
| 003_17.PNG | 003_13.PNG | True | 21 |
| 003_03.PNG | 003_06.PNG | True | 39 |
| 003_18.PNG | 003_17.PNG | True | 21 |
| 002_12.PNG | 002_13.PNG | True | 22 |
| 002_18.PNG | 002_11.PNG | True | 20 |
| 002_02.PNG | 002_04.PNG | True | 15 |
| 003_01.PNG | 003_05.PNG | True | 16 |
| 002_13.PNG | 002_14.PNG | True | 19 |
| 002_06.PNG | 002_15.PNG | True | 18 |
| 004_15.PNG | 004_23.PNG | True | 36 |
| 002_06.PNG | 002_23.PNG | True | 17 |
| 001_15.PNG | 001_08.PNG | True | 20 |
| 002_23.PNG | 002_16.PNG | True | 18 |
| 001_21.PNG | 001_22.PNG | True | 26 |
| 004_21.PNG | 004_09.PNG | True | 14 |
| 002_18.PNG | 002_11.PNG | True | 15 |
| 002_07.PNG | 002_15.PNG | True | 18 |
| 003_16.PNG | 003_23.PNG | True | 16 |
| 003_23.PNG | 003_17.PNG | True | 19 |
| 002_04.PNG | 002_11.PNG | True | 15 |
| 004_05.PNG | 004_18.PNG | True | 15 |
| 003_05.PNG | 003_12.PNG | True | 15 |
| 004_21.PNG | 004_02.PNG | True | 15 |
| 003_15.PNG | 003_10.PNG | True | 15 |
| 004_11.PNG | 004_16.PNG | True | 18 |
| 003_14.PNG | 003_01.PNG | True | 24 |
| 001_10.PNG | 001_16.PNG | True | 23 |

### Detailed test report

All 30 forged comparisons were correctly detected using a threshold of 0.8. All
30 genuine comparisons were accepted with a threshold of 6.0. The average
verification time was about 18.8 ms for forged pairs and 22.5 ms for genuine
pairs.

## Meta‑learning

Use the `sigver.metalearning.train` script to train a meta‑learner:

```bash
python -m sigver.metalearning.train --dataset-path <path/to/dataset.npz> \
    --pretrain-epochs <pretrain> --num-updates <gradient_steps> --num-rf <rand_forg> \
    --epochs <epochs> --num-sk-test <skilled_in_Dtest> --model <model>
```
`num-updates` specifies `K` in the paper, while `num-rf` controls how many random forgeries are used during adaptation.

## Citation

If you use this code, please cite:

1. Hafemann et al., *Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks* (2017).
2. Hafemann et al., *Characterizing and evaluating adversarial examples for Offline Handwritten Signature Verification* (2019).
3. Hafemann et al., *Meta-learning for fast classifier adaptation to new users of Signature Verification systems* (2019).

## License

The source code is released under the BSD 3‑Clause license. Models were trained on the GPDS dataset, which is restricted for non‑commercial use.

---

### Original project

This repository is based on the original [sigver](https://github.com/luizgh/sigver) project by Luiz G. Hafemann. The original README contains detailed explanations of the training procedure and dataset preparation which are reproduced here for convenience.
