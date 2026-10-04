"""Huấn luyện và so sánh các mô hình phát hiện spam.

Các mô hình (cùng dùng chung bộ đặc trưng TF-IDF + đặc trưng thủ công):
    nb        MultinomialNB
    lr        LogisticRegression
    svm       LinearSVC (calibrate để có xác suất)
    ensemble  Soft Voting của 3 mô hình trên

Mô hình tốt nhất được chọn theo F1 của lớp spam, tính bằng cross-validation
trên tập train (không nhìn tập test, tránh rò rỉ dữ liệu). Kết quả đánh giá cuối
cùng trên tập test được lưu vào models/metrics.json để giao diện hiển thị.

Cách dùng:
    python train.py                      # tự tải SMS Spam Collection (UCI); lỗi mạng thì dùng data/sample_spam.csv
    python train.py --data my_data.csv   # CSV có 2 cột: label (spam/ham), text
    python train.py --use-feedback       # thêm phản hồi 👍/👎 từ giao diện (logs/feedback.csv) vào dữ liệu train
"""
import argparse
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC

from preprocess import EXTRA_FEATURE_NAMES, clean_series, extra_features
from storage import FEEDBACK_FILE, read_feedback

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

MODEL_LABELS = {
    "nb": "MultinomialNB",
    "lr": "LogisticRegression",
    "svm": "LinearSVC (calibrated)",
    "ensemble": "Soft Voting (NB + LR + SVM)",
}
MODEL_NAMES = list(MODEL_LABELS)


# ----------------------------------------------------------------------------- dữ liệu
def download_uci_dataset() -> pd.DataFrame:
    """Tải SMS Spam Collection (UCI). File gốc là TSV: label \\t text."""
    resp = requests.get(UCI_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("SMSSpamCollection") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "sms_spam.csv", index=False)
    return df


def load_data(path: str | None) -> tuple[pd.DataFrame, str]:
    """Trả về (dataframe, tên nguồn dữ liệu)."""
    if path:
        return pd.read_csv(path), str(path)
    cached = DATA_DIR / "sms_spam.csv"
    if cached.exists():
        return pd.read_csv(cached), "SMS Spam Collection (UCI, cache)"
    try:
        print("Đang tải SMS Spam Collection từ UCI...")
        return download_uci_dataset(), "SMS Spam Collection (UCI)"
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Không tải được dataset ({exc}). Dùng data/sample_spam.csv (nhỏ, chỉ để demo).")
        return pd.read_csv(DATA_DIR / "sample_spam.csv"), "sample_spam.csv (demo)"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df[df["label"].isin(["spam", "ham"])].drop_duplicates(subset="text")


# ----------------------------------------------------------------------------- mô hình
def make_features() -> FeatureUnion:
    """Nhánh 1: làm sạch + TF-IDF (1-2 gram). Nhánh 2: đặc trưng thủ công (link, chữ HOA, dấu !...)."""
    text_branch = Pipeline([
        ("clean", FunctionTransformer(clean_series, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ])
    extra_branch = Pipeline([
        ("extra", FunctionTransformer(extra_features, validate=False)),
        ("scale", MinMaxScaler(clip=True)),  # đưa về [0, 1] để tương thích MultinomialNB
    ])
    return FeatureUnion([("text", text_branch), ("extra", extra_branch)])


def make_classifier(name: str, seed: int = 42):
    def nb():
        return MultinomialNB(alpha=0.1)

    def lr():
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)

    def svm():
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), cv=3)

    if name == "nb":
        return nb()
    if name == "lr":
        return lr()
    if name == "svm":
        return svm()
    if name == "ensemble":
        return VotingClassifier([("nb", nb()), ("lr", lr()), ("svm", svm())], voting="soft")
    raise ValueError(f"Model không hợp lệ: {name}")


def build_pipeline(name: str, seed: int = 42) -> Pipeline:
    """Pipeline nhận văn bản GỐC (chưa làm sạch) -> toàn bộ tiền xử lý nằm trong pipeline."""
    return Pipeline([("features", make_features()), ("clf", make_classifier(name, seed))])


def top_terms(lr_pipeline: Pipeline, k: int = 15) -> dict:
    """Từ có trọng số cao nhất theo Logistic Regression (để hiển thị trên UI).

    'spam'/'ham' chỉ chứa TỪ; các đặc trưng thủ công (số link, tỉ lệ chữ HOA...) nằm riêng ở 'extra'
    vì thang đo của chúng khác từ nên không so sánh trực tiếp được.
    """
    tfidf = lr_pipeline.named_steps["features"].named_transformers["text"].named_steps["tfidf"]
    words = list(tfidf.get_feature_names_out())
    coef = lr_pipeline.named_steps["clf"].coef_[0]
    word_coef, extra_coef = coef[:len(words)], coef[len(words):]
    order = np.argsort(word_coef)

    def pick(idx):
        return [{"term": str(words[i]), "weight": round(float(word_coef[i]), 4)} for i in idx]

    extra = sorted(
        ({"term": n, "weight": round(float(w), 4)} for n, w in zip(EXTRA_FEATURE_NAMES, extra_coef)),
        key=lambda d: d["weight"], reverse=True,
    )
    return {"spam": pick(order[::-1][:k]), "ham": pick(order[:k]), "extra": extra}


# ----------------------------------------------------------------------------- huấn luyện
def run_training(data_path: str | None = None, model_dir: Path = MODEL_DIR, seed: int = 42,
                 cv_folds: int = 5, verbose: bool = True, use_feedback: bool = False,
                 feedback_path: Path = FEEDBACK_FILE) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    model_dir = Path(model_dir)

    raw, source = load_data(data_path)
    n_feedback = 0
    if use_feedback:
        rows = read_feedback(feedback_path)
        if rows:
            fb = pd.DataFrame(rows).rename(columns={"correct_label": "label"})[["label", "text"]]
            n_feedback = len(fb)
            raw = pd.concat([raw[["label", "text"]], fb], ignore_index=True)
            source += f" + {n_feedback} phản hồi người dùng"
        else:
            log("[!] Không có phản hồi nào để dùng (logs/feedback.csv trống hoặc chưa có).")
    df = prepare_data(raw)
    counts = df["label"].value_counts()
    log(f"Nguồn: {source} | Số mẫu: {len(df)} | spam={counts.get('spam', 0)} ham={counts.get('ham', 0)}\n")

    X, y = df["text"], (df["label"] == "spam").astype(int)
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if can_stratify else None
    )

    n_splits = min(cv_folds, int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if n_splits >= 2 else None

    model_dir.mkdir(parents=True, exist_ok=True)
    results, fitted = {}, {}
    for name in MODEL_NAMES:
        cv_scores = (cross_val_score(build_pipeline(name, seed), X_train, y_train, cv=cv, scoring="f1")
                     if cv is not None else None)
        model = build_pipeline(name, seed).fit(X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        results[name] = {
            "label": MODEL_LABELS[name],
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4) if cv_scores is not None else None,
            "cv_f1_std": round(float(cv_scores.std()), 4) if cv_scores is not None else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        fitted[name] = model
        joblib.dump(model, model_dir / f"{name}.joblib")

    def selection_score(n: str) -> float:
        r = results[n]
        return r["cv_f1_mean"] if r["cv_f1_mean"] is not None else r["f1"]

    best = max(MODEL_NAMES, key=selection_score)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_metric": "cv_f1_spam" if cv is not None else "test_f1_spam",
        "best": best,
        "dataset": {
            "source": source, "n_samples": int(len(df)),
            "n_spam": int(counts.get("spam", 0)), "n_ham": int(counts.get("ham", 0)),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)), "cv_folds": n_splits,
            "n_feedback": n_feedback,
        },
        "models": results,
        "top_terms": top_terms(fitted["lr"]),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        table = pd.DataFrame(results).T[["cv_f1_mean", "accuracy", "precision", "recall", "f1"]]
        print("Kết quả trên tập test (cv_f1_mean = F1 spam khi cross-validation trên tập train):")
        print(table.to_string())
        for n, r in results.items():
            c = r["confusion_matrix"]
            print(f"  {n:<9} TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
        print(f"\nĐã lưu {len(MODEL_NAMES)} mô hình + metrics.json vào {model_dir}")
        print(f"Mô hình tốt nhất (theo {metrics['selection_metric']}): {best} - {MODEL_LABELS[best]}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Spam Email Detector")
    parser.add_argument("--data", default=None, help="Đường dẫn CSV (cột: label, text)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--use-feedback", action="store_true",
                        help="Thêm các phản hồi người dùng (logs/feedback.csv) vào dữ liệu huấn luyện")
    args = parser.parse_args()
    run_training(args.data, Path(args.model_dir), args.seed, args.cv_folds, use_feedback=args.use_feedback)


if __name__ == "__main__":
    main()
