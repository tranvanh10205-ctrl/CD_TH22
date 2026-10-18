# Speakify Local — SpeechOcean762 + TS-TCC
Responsive local practice web prototype inspired by the supplied mockup.

## Structure
- `app.py`: FastAPI endpoints and page
- `inference.py`: loads existing TS-TCC `base_Model` and train_linear checkpoint
- `audio_preprocess.py`: converts audio to (1,64,1536) log-Mel tensor
- `templates/index.html`: dashboard, lessons, recording, results, progress
- `static/css/styles.css`: responsive styling
- `static/js/app.js`: lesson navigation, browser recording, API calls, local progress

## Install/run
Put this folder inside your TS-TCC repository next to `models/` and `config_files/`.
Activate the existing venv from the TS-TCC root, then:
```powershell
python -m pip install -r .\speakify_local_web\requirements.txt
Copy-Item .\speakify_local_web\.env.example .\speakify_local_web\.env
# Edit .env: set TS_TCC_ROOT to the absolute TS-TCC root
python -m uvicorn speakify_local_web.app:app --reload
```
Open http://127.0.0.1:8000

## Important
Verify `audio_preprocess.py` against the exact script that produced `train.pt`; same tensor dimensions do not guarantee identical features. The current model returns experimental classes Low/Medium/High, not a validated score out of 100 or phoneme-level feedback. If checkpoint cannot load, `/api/health` explains why and prediction fails rather than fabricating a result. Browser recording commonly creates WebM; if decoding fails install FFmpeg or upload WAV.
