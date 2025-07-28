import os
import torch
from sigver.featurelearning.models.signet import SigNet

MODELS = ['signet', 'signet_f_lambda_0.95']

for name in MODELS:
    pth = os.path.join('models', f'{name}.pth')
    state_dict, _, _ = torch.load(pth, map_location='cpu')
    model = SigNet().eval()
    model.load_state_dict(state_dict)
    dummy = torch.randn(1, 1, 150, 220)
    onnx_path = os.path.join('models', f'{name}.onnx')
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['features'],
        opset_version=12,
    )
    print(f"Saved {onnx_path}")
