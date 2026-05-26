import os
import random
import rawpy
import numpy as np
import torch
from torch.utils.data import Dataset

class RAWDatasetFast(Dataset):
    """
    Турбо-датасет: один раз загружает всё в RAM, чтобы не ждать Google Диск.
    """
    def __init__(self, dark_dir, light_dir, patch_size=128, patches_per_image=100):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        valid_ext = ('.cr2', '.cr3', '.dng', '.nef', '.arw')
        
        all_files = os.listdir(dark_dir)
        self.image_names = sorted([f for f in all_files if f.lower().endswith(valid_ext)])
        
        self.data_cache = []
        
        for img_name in self.image_names:
            dark_path = os.path.join(dark_dir, img_name)
            light_path = os.path.join(light_dir, img_name)
            
            dark_tensor = self._load_and_pack(dark_path)
            light_tensor = self._load_and_pack(light_path)
            
            self.data_cache.append((dark_tensor, light_tensor))
            
        print("Данные в RAM")

    def __len__(self):
        return len(self.image_names) * self.patches_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.patches_per_image
        
        # Берем готовые тензоры прямо из памяти
        dark_tensor, light_tensor = self.data_cache[image_idx]

        h, w = dark_tensor.shape[1], dark_tensor.shape[2]
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        dark_patch = dark_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
        light_patch = light_tensor[:, y:y+self.patch_size, x:x+self.patch_size]
        # Подгоняем экспозицию темного патча под светлый, как в архивной версии
        ratio = light_patch.mean() / (dark_patch.mean() + 1e-5)
        ratio = torch.clamp(ratio, 1.0, 100.0)
        dark_patch = torch.clamp(dark_patch * ratio, 0.0, 1.0)

        # Случайные отражения для увеличения разнообразия данных (Аугментация)
        if random.random() > 0.5:
            dark_patch = torch.flip(dark_patch, [2])
            light_patch = torch.flip(light_patch, [2])
        if random.random() > 0.5:
            dark_patch = torch.flip(dark_patch, [1])
            light_patch = torch.flip(light_patch, [1])

        return dark_patch, light_patch

    def _load_and_pack(self, file_path):
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
            
            return torch.from_numpy(packed)