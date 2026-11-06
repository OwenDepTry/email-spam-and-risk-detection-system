"""Module dự đoán: dùng chung cho app.py (Streamlit), CLI và (sau này) API.

Các pipeline đã lưu nhận văn bản GỐC, nên không cần tiền xử lý thủ công ở đây.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_THRESHOLD = 0.5


# ----------------------------------------------------------------------------- nạp model
def load_metrics(model_dir: Path = MODEL_DIR) -> dict:
    path = Path(model_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy `python train.py` trước.")
    return json.loads(path.read_text(encoding="utf-8"))


def available_models(model_dir: Path = MODEL_DIR) -> list[str]:
    return list(load_metrics(model_dir)["models"])


def default_model_name(model_dir: Path = MODEL_DIR) -> str:
    return load_metrics(model_dir)["best"]


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int):
    return joblib.load(path)  # mtime nằm trong khóa cache -> train lại là tự nạp bản mới


def load_model(name: str | None = None, model_dir: Path = MODEL_DIR):
    model_dir = Path(model_dir)
    name = name or default_model_name(model_dir)
    path = model_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình '{name}'. Hãy chạy `python train.py` trước.")
    return _load_cached(str(path), path.stat().st_mtime_ns)


# ----------------------------------------------------------------------------- dự đoán
def spam_probabilities(texts, model_name: str | None = None, model_dir: Path = MODEL_DIR) -> np.ndarray:
    """Xác suất spam (0-1) cho từng văn bản."""
    model = load_model(model_name, model_dir)
    return model.predict_proba(list(texts))[:, 1]


def _to_result(proba_spam: float, threshold: float, model_name: str) -> dict:
    is_spam = proba_spam >= threshold
    return {
        "label": "Spam" if is_spam else "Ham",
        "confidence": round((proba_spam if is_spam else 1 - proba_spam) * 100, 2),
        "spam_probability": round(proba_spam * 100, 2),
        "model": model_name,
        "threshold": threshold,
    }


def predict_email(text: str, model_name: str | None = None, threshold: float = DEFAULT_THRESHOLD,
                  model_dir: Path = MODEL_DIR) -> dict:
    """Trả về {'label': 'Spam'|'Ham', 'confidence': 0-100, 'spam_probability': 0-100, 'model', 'threshold'}."""
    model_name = model_name or default_model_name(model_dir)
    proba = float(spam_probabilities([text], model_name, model_dir)[0])
    return _to_result(proba, threshold, model_name)


def predict_batch(texts, model_name: str | None = None, threshold: float = DEFAULT_THRESHOLD,
                  model_dir: Path = MODEL_DIR) -> list[dict]:
    texts = ["" if not isinstance(t, str) else t for t in texts]
    if not texts:
        return []
    model_name = model_name or default_model_name(model_dir)
    probas = spam_probabilities(texts, model_name, model_dir)
    return [_to_result(float(p), threshold, model_name) for p in probas]


def predict_all_models(text: str, threshold: float = DEFAULT_THRESHOLD, model_dir: Path = MODEL_DIR) -> list[dict]:
    """Kết quả của TẤT CẢ mô hình cho cùng một văn bản (để xem mức đồng thuận giữa các model)."""
    results = []
    for name, info in load_metrics(model_dir)["models"].items():
        proba = float(spam_probabilities([text], name, model_dir)[0])
        res = _to_result(proba, threshold, name)
        res["model_label"] = info["label"]
        results.append(res)
    return results


def explain_email(text: str, model_name: str | None = None, max_tokens: int = 80,
                  model_dir: Path = MODEL_DIR) -> list[dict]:
    """Giải thích bằng cách bỏ từng từ một (leave-one-out) và đo xác suất spam thay đổi bao nhiêu.

    Trả về danh sách {'token', 'impact'} theo đúng thứ tự trong văn bản.
    impact > 0: từ đó đẩy email về phía spam; impact < 0: đẩy về phía ham.
    Chỉ tính cho `max_tokens` từ đầu tiên để giữ thời gian phản hồi nhanh; các từ sau đó có
    impact = None (CHƯA phân tích), khác với impact = 0 (đã phân tích, không ảnh hưởng).
    """
    tokens = text.split()
    if not tokens:
        return []
    n = min(len(tokens), max_tokens)
    variants = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(n)]
    base = float(spam_probabilities([text], model_name, model_dir)[0])
    probs = spam_probabilities(variants, model_name, model_dir)
    impacts = [round(base - float(p), 4) for p in probs] + [None] * (len(tokens) - n)
    return [{"token": tok, "impact": imp} for tok, imp in zip(tokens, impacts)]


def evaluate_labels(true_labels, predicted_labels) -> dict | None:
    """Accuracy / precision / recall (lớp spam) CHỈ trên các dòng có nhãn thật hợp lệ.

    Nhãn hợp lệ: spam/ham hoặc 1/0 (không phân biệt hoa thường). Dòng thiếu nhãn bị bỏ khỏi MỌI chỉ số
    (trước đây precision vẫn đếm cả dự đoán spam ở dòng không có nhãn). Trả về None nếu không có dòng hợp lệ.
    """
    mapping = {"spam": "Spam", "1": "Spam", "1.0": "Spam", "ham": "Ham", "0": "Ham", "0.0": "Ham"}
    pairs = [(mapping.get(str(t).strip().lower()), p) for t, p in zip(true_labels, predicted_labels)]
    pairs = [(t, p) for t, p in pairs if t is not None]
    if not pairs:
        return None
    tp = sum(t == "Spam" and p == "Spam" for t, p in pairs)
    pred_spam = sum(p == "Spam" for _, p in pairs)
    true_spam = sum(t == "Spam" for t, _ in pairs)
    return {
        "n_labeled": len(pairs),
        "accuracy": sum(t == p for t, p in pairs) / len(pairs),
        "precision": tp / pred_spam if pred_spam else None,
        "recall": tp / true_spam if true_spam else None,
    }


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Phân loại một email/tin nhắn từ dòng lệnh")
    parser.add_argument("text", nargs="*", help="Nội dung cần kiểm tra (bỏ trống để nhập tay)")
    parser.add_argument("--model", default=None, help="nb | lr | svm | ensemble (mặc định: model tốt nhất)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--explain", action="store_true", help="Hiện các từ ảnh hưởng nhiều nhất")
    args = parser.parse_args()

    content = " ".join(args.text) or input("Nhập nội dung email: ")
    print(predict_email(content, args.model, args.threshold))
    if args.explain:
        scored = [d for d in explain_email(content, args.model) if d["impact"] is not None]
        top = sorted(scored, key=lambda d: abs(d["impact"]), reverse=True)[:8]
        for item in top:
            print(f"  {item['impact']:+.3f}  {item['token']}")


if __name__ == "__main__":
    main()
