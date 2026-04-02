import os
import tempfile
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from raw_enhancement.models.my_model import LightweightUNet
from raw_enhancement.utils.processing import process_with_native_isp

from fastapi.responses import RedirectResponse
app = FastAPI()


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

print("Загрузка нейросети...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightUNet().to(device)

model_weights_dir = os.path.join(os.path.dirname(__file__), "src", "raw_enhancement", "models", "model_weights")
weight_name = "model_epoch_200.pth"
model_path = os.path.join(model_weights_dir, weight_name)
if not os.path.exists(model_path):
    fallback_path = os.path.join(os.path.dirname(__file__), weight_name)
    if os.path.exists(fallback_path):
        model_path = fallback_path
    else:
        raise FileNotFoundError(f"Model file '{weight_name}' not found in {model_weights_dir} or project root.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Бэкенд готов к работе!")

@app.post("/api/process")
async def process_raw(
    file: UploadFile = File(...),
    strength: float = Form(100.0),
    exposure: float = Form(0.0),
    temp: float = Form(0.0)
):
    import rawpy

    temp_raw = f"temp_{file.filename}"
    with open(temp_raw, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    ai_image = process_with_native_isp(temp_raw, model, device).astype(np.float32)

    with rawpy.imread(temp_raw) as raw:
        orig_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True).astype(np.float32)

    h = min(ai_image.shape[0], orig_image.shape[0])
    w = min(ai_image.shape[1], orig_image.shape[1])
    ai_image = ai_image[:h, :w]
    orig_image = orig_image[:h, :w]

    alpha = strength / 100.0
    final_image = (orig_image * (1.0 - alpha)) + (ai_image * alpha)

    if exposure != 0.0:
        expo_factor = 2.0 ** exposure
        final_image = final_image * expo_factor

    if temp != 0.0:
        temp_shift = temp * 0.5 
        final_image[:, :, 0] += temp_shift
        final_image[:, :, 2] -= temp_shift

    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    output_filename = "result_output.jpg"
    img_pil = Image.fromarray(final_image)
    img_pil.save(output_filename, quality=95)

    os.remove(temp_raw)
    return FileResponse(output_filename, media_type="image/jpeg")