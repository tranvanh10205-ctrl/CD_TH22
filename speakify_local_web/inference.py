from pathlib import Path
import os, sys
import numpy as np
import torch
from dotenv import load_dotenv
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
ROOT = Path(os.getenv("TS_TCC_ROOT", str(BASE.parent))).expanduser().resolve()
CKPT = Path(os.getenv("CHECKPOINT_PATH", "experiments_logs/SpeechOcean762_TS-TCC/linear_bn_fixed/train_linear_seed_0/saved_models/ckp_best.pt"))
if not CKPT.is_absolute(): CKPT = ROOT / CKPT
dv = os.getenv("DEVICE", "auto").lower()
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dv == "auto" else dv)
CLASSES = ["Low", "Medium", "High"]
_model = None
_error = None

def _load():
    global _model, _error
    if _model is not None: return _model
    try:
        if not ROOT.exists(): raise FileNotFoundError(f"TS_TCC_ROOT không tồn tại: {ROOT}")
        if not CKPT.exists(): raise FileNotFoundError(f"Không tìm thấy checkpoint: {CKPT}")
        sys.path.insert(0, str(ROOT))
        from config_files.SpeechOcean762_Configs import Config
        from models.model import base_Model
        cfg = Config(); cfg.input_channels = 64; cfg.num_classes = 3
        model = base_Model(cfg).to(DEVICE)
        try: state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        except TypeError: state = torch.load(CKPT, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
        model.load_state_dict(state, strict=True); model.eval()
        _model = model; _error = None
    except Exception as e: _error = str(e)
    return _model

def model_status():
    ready = _load() is not None
    return {"ready": ready, "checkpoint": str(CKPT), "device": str(DEVICE),
            "message": "Mô hình đã nạp." if ready else f"Chưa nạp được mô hình: {_error}",
            "classes": CLASSES}

def predict_audio(path):
    model = _load()
    if model is None: raise RuntimeError("Mô hình chưa sẵn sàng. Kiểm tra .env và checkpoint. " + str(_error))
    from .audio_preprocess import wav_to_tensor
    x = wav_to_tensor(path).to(DEVICE)
    with torch.inference_mode():
        out = model(x); logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    i = int(np.argmax(probs))
    return {"ok": True, "predicted_class_id": i, "predicted_class": CLASSES[i],
            "confidence": round(float(probs[i]), 4),
            "class_probabilities": {CLASSES[j]: round(float(probs[j]), 4) for j in range(3)},
            "interpretation": f"Mô hình xếp mẫu vào nhóm {CLASSES[i]}.",
            "limitations": "Đây là phân loại 3 lớp thử nghiệm, không phải điểm phát âm chuẩn 0–100 hoặc nhận xét lỗi từng âm."}
