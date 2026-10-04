"""Lưu phản hồi (feedback) của người dùng ra CSV.

Dữ liệu này dùng để retrain sau này. Khi nâng cấp lên database (SQLite/PostgreSQL),
chỉ cần thay phần ghi/đọc trong file này, các nơi gọi không đổi.
"""
import csv
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
FEEDBACK_FILE = LOG_DIR / "feedback.csv"
FIELDS = ["timestamp", "model", "predicted", "correct_label", "spam_probability", "text"]


def log_feedback(text: str, model: str, predicted: str, correct_label: str,
                 spam_probability: float, path: Path = FEEDBACK_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "predicted": predicted,
            "correct_label": correct_label,
            "spam_probability": spam_probability,
            "text": text,
        })


def read_feedback(path: Path = FEEDBACK_FILE) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
