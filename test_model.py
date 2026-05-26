import os
import torch
import torch.nn as nn
import numpy as np
import rawpy
import pandas as pd
from pytorch_msssim import ssim
from tqdm.auto import tqdm

dark_dir = '/content/drive/MyDrive/VKR_Data/raw_dark_test'
light_dir = '/content/drive/MyDrive/VKR_Data/raw_light_test'
checkpoint_path = '/content/drive/MyDrive/VKR_Checkpoints/model_epoch_200.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_pack_for_eval(file_path):
    """Загрузка RAW и упаковка в 4 канала (RGGB/BGGR)"""
    with rawpy.imread(file_path) as raw:
        raw_data = raw.raw_image_visible.astype(np.float32)
        black = raw.black_level_per_channel[0]
        white = raw.white_level

        raw_norm = np.clip((raw_data - black) / (white - black), 0.0, 1.0)
        h, w = raw_norm.shape

        packed = np.zeros((4, h // 2, w // 2), dtype=np.float32)
        packed[0, :, :] = raw_norm[0::2, 0::2]
        packed[1, :, :] = raw_norm[0::2, 1::2]
        packed[2, :, :] = raw_norm[1::2, 0::2]
        packed[3, :, :] = raw_norm[1::2, 1::2]

        tensor = torch.from_numpy(packed).unsqueeze(0)
        _, _, curr_h, curr_w = tensor.shape
        safe_h = (curr_h // 16) * 16
        safe_w = (curr_w // 16) * 16

        return tensor[:, :, :safe_h, :safe_w]

def calculate_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse)


model = LightweightUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

valid_ext = ('.cr2', '.cr3', '.dng')
image_names = sorted([f for f in os.listdir(dark_dir) if f.lower().endswith(valid_ext)])

criterion_mae = nn.L1Loss()
results = []


with torch.no_grad(): 
    for idx, img_name in enumerate(tqdm(image_names)):
        dark_path = os.path.join(dark_dir, img_name)
        light_path = os.path.join(light_dir, img_name)

        dark_tensor = load_and_pack_for_eval(dark_path)
        light_tensor = load_and_pack_for_eval(light_path)


        ratio = light_tensor.mean() / (dark_tensor.mean() + 1e-5)
        ratio = torch.clamp(ratio, 1.0, 100.0)
        dark_tensor = torch.clamp(dark_tensor * ratio, 0.0, 1.0)

        crop_h, crop_w = 1024, 1024 
        _, _, h, w = dark_tensor.shape
        start_y = max(0, (h - crop_h) // 2)
        start_x = max(0, (w - crop_w) // 2)

        dark_crop = dark_tensor[:, :, start_y:start_y+crop_h, start_x:start_x+crop_w].to(device)
        light_crop = light_tensor[:, :, start_y:start_y+crop_h, start_x:start_x+crop_w].to(device)

        outputs = model(dark_crop)
        outputs = torch.clamp(outputs, 0.0, 1.0)

        mae = criterion_mae(outputs, light_crop).item()
        psnr = calculate_psnr(outputs, light_crop).item()
        ssim_val = ssim(outputs, light_crop, data_range=1.0, size_average=True).item()

        ext = os.path.splitext(img_name)[1].upper().replace('.', '')

        results.append({
            '№': idx + 1,
            'Формат': ext,
            'MAE': round(mae, 4),
            'PSNR': round(psnr, 2),
            'SSIM': round(ssim_val, 4),
        })

        torch.cuda.empty_cache()

df = pd.DataFrame(results)

mean_row = pd.DataFrame([{
    '№ пары': 'Среднее',
    'Формат': '-',
    'MAE': round(df['MAE'].mean(), 4),
    'PSNR ↑': round(df['PSNR'].mean(), 2),
    'SSIM ↑': round(df['SSIM'].mean(), 4)
}])

df = pd.concat([df, mean_row], ignore_index=True)


csv_path = '/content/drive/MyDrive/VKR_Data/test_metrics_table.csv'
df.to_csv(csv_path, sep=';', index=False, encoding='utf-8-sig')

display(df)