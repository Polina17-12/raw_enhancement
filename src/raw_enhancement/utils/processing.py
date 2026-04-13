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
    """Placeholder for native ISP rendering (full implementation will be added in next commit)."""
    raise NotImplementedError("process_with_native_isp placeholder — full implementation in next commit")
