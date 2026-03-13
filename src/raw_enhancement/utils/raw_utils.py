import rawpy
import numpy as np


def load_and_pack_raw(file_path):

    with rawpy.imread(file_path) as raw:
        raw_data = raw.raw_image_visible.astype(np.float32)
        black_level = raw.black_level_per_channel[0]
        white_level = raw.white_level
        raw_normalized = np.maximum(raw_data - black_level, 0) / (white_level - black_level)

        h, w = raw_normalized.shape
        packed_raw = np.zeros((h // 2, w // 2, 4), dtype=np.float32)
        packed_raw[:, :, 0] = raw_normalized[0::2, 0::2]
        packed_raw[:, :, 1] = raw_normalized[0::2, 1::2]
        packed_raw[:, :, 2] = raw_normalized[1::2, 0::2]
        packed_raw[:, :, 3] = raw_normalized[1::2, 1::2]

        return packed_raw, raw_normalized


def simple_demosaic_baseline(file_path):
    """Базова конвертация RAW в sRGB"""
    with rawpy.imread(file_path) as raw:
        rgb_image = raw.postprocess(half_size=True, use_camera_wb=True)
        return rgb_image
