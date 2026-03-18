"""Простой скрипт инференса, использующий пакетную структуру.
Запускается как модуль: `python -m raw_enhancement.inference.inference`
"""
import os
import pathlib
import torch
import matplotlib.pyplot as plt
from raw_enhancement.models.my_model import LightweightUNet
from raw_enhancement.utils.processing import load_full_raw, unpack_to_rgb_hybrid

def main(raw_path: str, out_path: str = "result.jpg", model_path: str = None):
    device = torch.device("cpu")

    model = LightweightUNet().to(device)

    if model_path is None:
        here = pathlib.Path(__file__).resolve().parent
        model_path = here.parent / 'models' / 'model_weights' / 'model_epoch_30.pth'

    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    input_tensor = load_full_raw(raw_path).to(device)

    with torch.no_grad():
        processed_tensor = model(input_tensor)

    final_image = unpack_to_rgb_hybrid(processed_tensor[0], wb_red=2.4, wb_blue=1.8)

    plt.imsave(out_path, final_image)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('raw', help='Path to RAW file')
    p.add_argument('--out', default='result.jpg')
    p.add_argument('--model', default=None)
    args = p.parse_args()
    main(args.raw, args.out, args.model)
