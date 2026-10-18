from pathlib import Path
import os, tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .inference import model_status, predict_audio
from .language_check import check_english

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
app = FastAPI(title="Speakify Local")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/api/health")
async def health():
    return model_status()

from fastapi import HTTPException

# Đảm bảo check_english đã được định nghĩa
# hoặc import từ module chứa hàm này.

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"
}
import traceback
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Các bước kiểm tra định dạng file của bạn giữ nguyên
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Định dạng file không được hỗ trợ."
        )

    path = None

    try:
        # Ghi file upload vào đường dẫn tạm
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:
            temp.write(await file.read())
            path = temp.name

        # ===== CHÈN ĐOẠN KIỂM TRA TIẾNG ANH TẠI ĐÂY =====
        check = check_english(path)

        if not check["accepted"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": check["message"],
                    "detected_language": check["language"],
                    "language_probability": check["language_probability"],
                    "transcript": check["transcript"],
                },
            )

        # Chỉ chạy TS-TCC nếu đầu vào được chấp nhận
        result = predict_audio(path)

        result["transcript"] = check["transcript"]
        result["detected_language"] = check["language"]
        result["language_probability"] = check["language_probability"]

        return result

    except HTTPException:
        # Giữ nguyên mã lỗi 422 hoặc các lỗi HTTP khác
        raise

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        # Xóa file tạm kể cả khi kiểm tra ngôn ngữ thất bại
        if path and os.path.exists(path):
            os.remove(path)
