import os
import tempfile
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import sys
import uuid
import datetime

# Импорты для работы с PostgreSQL через SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, desc
from sqlalchemy.orm import declarative_base, sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from raw_enhancement.models.my_model import LightweightUNet
from raw_enhancement.utils.processing import process_with_native_isp

app = FastAPI()

# Конфигурация подключения к базе данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres_user:postgres_password@localhost:5432/history_db")
HISTORY_DIR = os.path.join("static", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Настройка фабрики сессий SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Декларативная модель таблицы истории для SQLAlchemy
class DBHistoryItem(Base):
    __tablename__ = "history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    filename = Column(String)
    output_img = Column(String)
    strength = Column(Float)
    exposure = Column(Float)
    temp = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Инициализация таблиц PostgreSQL
def init_db():
    Base.metadata.create_all(bind=engine)

init_db()

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

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
    temp: float = Form(0.0),
    session_id: str = Form("anonymous")
):
    import rawpy

    temp_raw = f"temp_{uuid.uuid4().hex}_{file.filename}"
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

    unique_id = uuid.uuid4().hex[:8]
    unique_filename = f"result_{unique_id}.png"
    orig_filename = f"orig_{unique_id}.png" 
    
    output_path = os.path.join(HISTORY_DIR, unique_filename)
    orig_output_path = os.path.join(HISTORY_DIR, orig_filename)
    
    img_pil = Image.fromarray(final_image)
    img_pil.save(output_path, format="PNG")
    
    orig_image_uint8 = np.clip(orig_image, 0, 255).astype(np.uint8)
    Image.fromarray(orig_image_uint8).save(orig_output_path, format="PNG")
    
    os.remove(temp_raw)

    # Сохранение записи через ORM-сессию SQLAlchemy
    db = SessionLocal()
    try:
        db_item = DBHistoryItem(
            session_id=session_id,
            filename=file.filename,
            output_img=unique_filename,
            strength=strength,
            exposure=exposure,
            temp=temp
        )
        db.add(db_item)
        db.commit()
    finally:
        db.close()

    return {
        "processed": f"/ui/history/{unique_filename}",
        "original": f"/ui/history/{orig_filename}"
    }

@app.get("/api/history")
async def get_history(session_id: str = "anonymous"):
    db = SessionLocal()
    try:
        rows = db.query(DBHistoryItem)\
                 .filter(DBHistoryItem.session_id == session_id)\
                 .order_by(desc(DBHistoryItem.id))\
                 .all()
    finally:
        db.close()
    
    return [
        {
            "id": item.id,
            "filename": item.filename,
            "output_img": f"/ui/history/{item.output_img}",
            "strength": item.strength,
            "exposure": item.exposure,
            "temp": item.temp
        } for item in rows
    ]
