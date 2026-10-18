
import os
from faster_whisper import WhisperModel

_asr = None


def _get_asr():
    global _asr
    if _asr is None:
        _asr = WhisperModel(
            os.getenv("WHISPER_MODEL", "tiny"),
            device="cpu",
            compute_type="int8",
        )
    return _asr


def check_english(audio_path):
    segments, info = _get_asr().transcribe(
        audio_path,
        beam_size=3,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    language = info.language
    probability = info.language_probability

    # Ngưỡng ban đầu để thử nghiệm, cần đánh giá trên dữ liệu thực tế.
    if language != "en" or probability < 0.80:
        return {
            "accepted": False,
            "language": language,
            "language_probability": round(probability, 3),
            "transcript": text,
            "message": (
                "Chưa xác minh được đây là tiếng Anh. "
                "Hãy ghi âm lại một câu tiếng Anh rõ ràng."
            ),
        }

    if len(text.split()) < 3:
        return {
            "accepted": False,
            "language": language,
            "language_probability": round(probability, 3),
            "transcript": text,
            "message": "Câu nói quá ngắn. Hãy đọc một câu tiếng Anh dài hơn.",
        }

    return {
        "accepted": True,
        "language": language,
        "language_probability": round(probability, 3),
        "transcript": text,
        "message": "Đã nhận diện đầu vào là tiếng Anh.",
    }
