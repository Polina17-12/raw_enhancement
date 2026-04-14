import rawpy
import numpy as np
import torch
from skimage import exposure, color



def load_full_raw(file_path):
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
        _, _, current_h, current_w = tensor.shape
        safe_h = (current_h // 16) * 16
        safe_w = (current_w // 16) * 16

        return tensor[:, :, :safe_h, :safe_w]

def unpack_to_rgb_hybrid(tensor, wb_red=2.4, wb_blue=1.8):
    r = tensor[0].cpu().numpy() * wb_red
    g = ((tensor[1] + tensor[2]) / 2.0).cpu().numpy()
    b = tensor[3].cpu().numpy() * wb_blue

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    hsv = color.rgb2hsv(rgb)
    hsv[:, :, 2] = exposure.equalize_adapthist(hsv[:, :, 2], clip_limit=0.02)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0.0, 1.0)
    
    rgb_clahe = color.hsv2rgb(hsv)
    rgb_final = rgb_clahe ** (1/2.2)

    return np.clip(rgb_final, 0.0, 1.0)


def process_with_native_isp(raw_path, model, device="cpu"):
    """Run model on packed RAW and render result through camera-native ISP."""
    # local import to avoid hard dependency at module import time
    import cv2

    with rawpy.imread(raw_path) as raw:
        raw_data = raw.raw_image_visible.astype(np.float32)
        black = raw.black_level_per_channel[0]
        white = raw.white_level
        raw_norm = np.clip((raw_data - black) / (white - black), 0.0, 1.0)

        ratio = 0.05 / (np.mean(raw_norm) + 1e-5)
        ratio = np.clip(ratio, 1.0, 100.0)
        raw_norm = np.clip(raw_norm * ratio, 0.0, 1.0)

        h, w = raw_norm.shape
        packed = np.zeros((4, h // 2, w // 2), dtype=np.float32)
        packed[0, :, :] = raw_norm[0::2, 0::2]
        packed[1, :, :] = raw_norm[0::2, 1::2]
        packed[2, :, :] = raw_norm[1::2, 0::2]
        packed[3, :, :] = raw_norm[1::2, 1::2]

        tensor = torch.from_numpy(packed).unsqueeze(0).to(device)

        _, _, curr_h, curr_w = tensor.shape
        safe_h = (curr_h // 16) * 16
        safe_w = (curr_w // 16) * 16
        tensor = tensor[:, :, :safe_h, :safe_w]

        with torch.no_grad():
            out_tensor = model(tensor)[0].cpu().numpy()

        out_tensor = np.clip(out_tensor, 0.0, 1.0)

        unpacked_norm = np.zeros((safe_h * 2, safe_w * 2), dtype=np.float32)
        unpacked_norm[0::2, 0::2] = out_tensor[0]
        unpacked_norm[0::2, 1::2] = out_tensor[1]
        unpacked_norm[1::2, 0::2] = out_tensor[2]
        unpacked_norm[1::2, 1::2] = out_tensor[3]

        unpacked_raw = unpacked_norm * (white - black) + black
        unpacked_raw = np.clip(unpacked_raw, black, white)

        try:
            raw.raw_image_visible.flags.writeable = True
        except:
            pass

        raw.raw_image_visible[:safe_h*2, :safe_w*2] = unpacked_raw.astype(raw.raw_image_visible.dtype)

        rgb_image = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=8,
        )

        rgb_image = rgb_image[:safe_h*2, :safe_w*2, :]

        return rgb_image
