"""Huấn luyện và so sánh các mô hình phát hiện spam.

Các mô hình (cùng dùng chung bộ đặc trưng TF-IDF + đặc trưng thủ công):
    nb        MultinomialNB
    lr        LogisticRegression
    svm       LinearSVC (calibrate để có xác suất)
    ensemble  Soft Voting của 3 mô hình trên

Mô hình tốt nhất được chọn theo F1 của lớp spam, tính bằng cross-validation
trên tập train (không nhìn tập test, tránh rò rỉ dữ liệu). Kết quả đánh giá cuối
cùng trên tập test được lưu vào models/metrics.json để giao diện hiển thị.

Chống rò rỉ dữ liệu:
    - Chia train/test và các fold CV THEO NHÓM: tin giống nhau sau làm sạch (preprocess.group_key)
      luôn nằm cùng một phía.
    - Tập test chỉ được chia từ dữ liệu gốc (không có phản hồi) và được lưu vào
      <model-dir>/test_split.json; các lần train sau dùng lại đúng tập test đó.
    - Phản hồi người dùng chỉ được đưa vào tập train; phản hồi trùng với một tin trong tập test
      bị bỏ qua (và được đếm lại) để không làm thay đổi tập test.

Cách dùng:
    python train.py                      # tự tải SMS Spam Collection (UCI); lỗi mạng thì dùng data/sample_spam.csv
    python train.py --data my_data.csv   # CSV có 2 cột: label (spam/ham), text
    python train.py --use-feedback       # thêm phản hồi 👍/👎 từ giao diện (logs/feedback.csv) vào dữ liệu train
"""
import argparse
import hashlib
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC

from preprocess import EXTRA_FEATURE_NAMES, clean_series, extra_features, group_key, normalize_whitespace
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
    """Chuẩn hóa nhãn, bỏ dòng thiếu và bỏ trùng theo văn bản đã chuẩn hóa khoảng trắng.

    Chỉ bỏ trùng khi hai tin giống hệt nhau (sau khi gộp khoảng trắng). Các tin chỉ giống nhau
    sau làm sạch được giữ lại nhưng gom vào cùng nhóm khi chia dữ liệu (xem split_train_test).
    """
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["text"].map(normalize_whitespace) != ""]
    df = df.loc[~df["text"].map(normalize_whitespace).duplicated()]
    return df[["label", "text"]].reset_index(drop=True)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Dấu vân tay của dữ liệu gốc: đổi dữ liệu thì tập test được chia lại."""
    rows = sorted(f"{lbl}\t{normalize_whitespace(t)}" for lbl, t in zip(df["label"], df["text"]))
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def split_train_test(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2,
                     split_file: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Chia theo nhóm, phân tầng theo nhãn. Nếu split_file đã có và khớp dữ liệu thì dùng lại tập test cũ.

    Trả về (train_df, test_df, info). Cả hai có thêm cột 'group'.
    """
    df = df.copy()
    df["group"] = df["text"].map(group_key)
    fingerprint = dataset_fingerprint(df)
    info = {"fingerprint": fingerprint, "reused": False}

    saved = None
    if split_file is not None and Path(split_file).exists():
        saved = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            saved = None  # dữ liệu gốc đã đổi -> chia lại

    if saved is not None:
        test_groups = set(saved["test_groups"])
        info["reused"] = True
    else:
        y = (df["label"] == "spam").astype(int)
        n_splits = max(2, round(1 / test_size))
        if y.value_counts().min() >= n_splits:
            # Lấy 1 fold của StratifiedGroupKFold làm test: vừa theo nhóm vừa giữ tỉ lệ spam/ham.
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            _, test_idx = next(sgkf.split(df, y, groups=df["group"]))
        else:  # dữ liệu quá nhỏ để phân tầng
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            _, test_idx = next(gss.split(df, y, groups=df["group"]))
        test_groups = set(df["group"].iloc[test_idx])
        if split_file is not None:
            Path(split_file).parent.mkdir(parents=True, exist_ok=True)
            Path(split_file).write_text(json.dumps({
                "fingerprint": fingerprint, "seed": seed, "test_size": test_size,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "test_groups": sorted(test_groups),
            }, indent=1), encoding="utf-8")

    is_test = df["group"].isin(test_groups)
    return df[~is_test].copy(), df[is_test].copy(), info


def merge_feedback(train_df: pd.DataFrame, test_df: pd.DataFrame, feedback: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Đưa phản hồi vào tập TRAIN.

    - Nhiều phản hồi cho cùng một tin: lấy phản hồi mới nhất.
    - Phản hồi trùng nhóm với một tin trong tập test: bỏ qua, để tập test không đổi.
    - Phản hồi trùng nội dung với một tin trong train: phản hồi thay thế nhãn gốc.
    """
    stats = {"n_feedback": int(len(feedback)), "n_feedback_used": 0, "n_feedback_in_test": 0,
             "n_label_overrides": 0}
    if feedback.empty:
        return train_df, stats
    # prepare_data giữ bản đầu tiên -> đảo thứ tự để giữ phản hồi MỚI NHẤT
    fb = feedback.copy()
    if "timestamp" in fb:
        fb = fb.sort_values("timestamp", kind="stable")
    fb = fb.iloc[::-1]
    fb = prepare_data(fb)
    fb["group"] = fb["text"].map(group_key)

    in_test = fb["group"].isin(set(test_df["group"]))
    stats["n_feedback_in_test"] = int(in_test.sum())
    fb = fb[~in_test]

    fb_norm = dict(zip(fb["text"].map(normalize_whitespace), fb["label"]))
    train_norm = train_df["text"].map(normalize_whitespace)
    replaced = train_norm.isin(fb_norm)
    stats["n_label_overrides"] = int(sum(fb_norm[t] != lbl for t, lbl in
                                         zip(train_norm[replaced], train_df.loc[replaced, "label"])))
    stats["n_feedback_used"] = int(len(fb))
    merged = pd.concat([train_df[~replaced], fb], ignore_index=True)
    return merged, stats


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
                 feedback_path: Path = FEEDBACK_FILE, resplit: bool = False) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    split_file = model_dir / "test_split.json"
    if resplit and split_file.exists():
        split_file.unlink()

    raw, source = load_data(data_path)
    base = prepare_data(raw)
    train_df, test_df, split_info = split_train_test(base, seed, split_file=split_file)
    log(("Dùng lại tập test đã khóa" if split_info["reused"] else "Đã chia và khóa tập test mới")
        + f" ({split_file.name}, dữ liệu {split_info['fingerprint']})")

    fb_stats = {"n_feedback": 0, "n_feedback_used": 0, "n_feedback_in_test": 0, "n_label_overrides": 0}
    if use_feedback:
        rows = read_feedback(feedback_path)
        if rows:
            fb = pd.DataFrame(rows).rename(columns={"correct_label": "label"})
            train_df, fb_stats = merge_feedback(train_df, test_df, fb)
            source += f" + {fb_stats['n_feedback_used']} phản hồi người dùng (chỉ vào train)"
            if fb_stats["n_feedback_in_test"]:
                log(f"[!] Bỏ qua {fb_stats['n_feedback_in_test']} phản hồi trùng với tin trong tập test.")
        else:
            log("[!] Không có phản hồi nào để dùng (logs/feedback.csv trống hoặc chưa có).")

    df = pd.concat([train_df, test_df], ignore_index=True)
    counts = df["label"].value_counts()
    log(f"Nguồn: {source} | Số mẫu: {len(df)} | spam={counts.get('spam', 0)} ham={counts.get('ham', 0)}\n")

    X_train, y_train = train_df["text"], (train_df["label"] == "spam").astype(int)
    X_test, y_test = test_df["text"], (test_df["label"] == "spam").astype(int)
    groups_train = train_df["group"]

    n_splits = min(cv_folds, int(y_train.value_counts().min()), int(groups_train.nunique()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed) if n_splits >= 2 else None

    results, fitted = {}, {}
    for name in MODEL_NAMES:
        cv_scores = (cross_val_score(build_pipeline(name, seed), X_train, y_train, groups=groups_train,
                                     cv=cv, scoring="f1")
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
            "split": "group (StratifiedGroupKFold), tập test khóa trong test_split.json",
            "split_fingerprint": split_info["fingerprint"],
            **fb_stats,
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
                        help="Thêm các phản hồi người dùng (logs/feedback.csv) vào tập train (không đụng tập test)")
    parser.add_argument("--resplit", action="store_true",
                        help="Bỏ tập test đã khóa và chia lại (kết quả sẽ không so sánh được với các lần trước)")
    args = parser.parse_args()
    run_training(args.data, Path(args.model_dir), args.seed, args.cv_folds, use_feedback=args.use_feedback,
                 resplit=args.resplit)


if __name__ == "__main__":
    main()
