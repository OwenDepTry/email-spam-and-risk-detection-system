"""Test hồi quy cho các lỗi đã tái hiện khi review (rò rỉ train/test, feedback, metric, URL, giải thích).

Chạy: pytest
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui_components as ui  # noqa: E402
from predict import evaluate_labels, explain_email  # noqa: E402
from preprocess import group_key  # noqa: E402
from signals import detect_signals, is_shortener  # noqa: E402
from storage import log_feedback  # noqa: E402
from train import merge_feedback, prepare_data, run_training, split_train_test  # noqa: E402

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_spam.csv"


def _toy_df(n_groups: int = 30) -> pd.DataFrame:
    """Dữ liệu nhỏ có nhiều cặp gần trùng (khác số, chữ hoa, khoảng trắng)."""
    rows = []
    for i in range(n_groups):
        label = "spam" if i % 3 == 0 else "ham"
        word = f"topic{chr(97 + i % 26)}{chr(97 + i // 26)}"
        base = f"{'Claim your prize' if label == 'spam' else 'See you at'} {word} call 0123{i}"
        rows += [{"label": label, "text": base},
                 {"label": label, "text": base.upper().replace("0123", "0999") + "  "}]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 1. rò rỉ train/test
def test_prepare_data_drops_whitespace_only_duplicates():
    df = pd.DataFrame({"label": ["ham", "ham", "spam"], "text": ["hello there", "  hello   there ", "win"]})
    assert len(prepare_data(df)) == 2


def test_near_duplicates_never_cross_train_test(tmp_path):
    train_df, test_df, _ = split_train_test(prepare_data(_toy_df()), split_file=tmp_path / "s.json")
    assert len(test_df) > 0 and len(train_df) > 0
    assert not set(train_df["group"]) & set(test_df["group"])
    assert group_key("Call 0123 NOW!") == group_key("call 9999 now")


def test_locked_test_set_is_reused(tmp_path):
    split_file = tmp_path / "s.json"
    df = prepare_data(_toy_df())
    _, test1, info1 = split_train_test(df, seed=1, split_file=split_file)
    _, test2, info2 = split_train_test(df, seed=999, split_file=split_file)  # seed khác vẫn ra cùng tập test
    assert not info1["reused"] and info2["reused"]
    assert set(test1["text"]) == set(test2["text"])


def test_changed_base_data_creates_new_split(tmp_path):
    split_file = tmp_path / "s.json"
    df = prepare_data(_toy_df())
    _, _, info1 = split_train_test(df, split_file=split_file)
    _, _, info2 = split_train_test(df.iloc[2:], split_file=split_file)
    assert info1["fingerprint"] != info2["fingerprint"] and not info2["reused"]


# ----------------------------------------------------------------- 2. feedback
def test_feedback_overrides_original_label_and_latest_wins():
    train_df, test_df, _ = split_train_test(prepare_data(_toy_df()))
    target = train_df[train_df["label"] == "spam"]["text"].iloc[0]
    fb = pd.DataFrame([
        {"timestamp": "2026-01-02T00:00:00", "label": "Ham", "text": target + " "},
        {"timestamp": "2026-01-01T00:00:00", "label": "Spam", "text": target},
    ])
    merged, stats = merge_feedback(train_df, test_df, fb)
    labels = merged.loc[merged["text"].str.strip() == target.strip(), "label"].tolist()
    assert labels == ["ham"]
    assert stats["n_label_overrides"] == 1 and len(merged) == len(train_df)


def test_feedback_matching_test_set_is_not_used():
    train_df, test_df, _ = split_train_test(prepare_data(_toy_df()))
    fb = pd.DataFrame([{"label": "spam", "text": test_df["text"].iloc[0]}])
    merged, stats = merge_feedback(train_df, test_df, fb)
    assert stats["n_feedback_in_test"] == 1 and stats["n_feedback_used"] == 0
    assert len(merged) == len(train_df)


def test_training_with_feedback_keeps_same_test_set(tmp_path):
    model_dir = tmp_path / "m"
    before = run_training(str(SAMPLE_CSV), model_dir, verbose=False)
    split_before = json.loads((model_dir / "test_split.json").read_text(encoding="utf-8"))

    fb = tmp_path / "fb.csv"
    log_feedback("totally new lottery jackpot click to collect", "nb", "Ham", "Spam", 10.0, fb)
    log_feedback("are we still on for dinner on friday", "nb", "Spam", "Ham", 90.0, fb)
    after = run_training(str(SAMPLE_CSV), model_dir, use_feedback=True, feedback_path=fb, verbose=False)
    split_after = json.loads((model_dir / "test_split.json").read_text(encoding="utf-8"))

    assert split_before == split_after
    assert after["dataset"]["n_test"] == before["dataset"]["n_test"]
    assert after["dataset"]["n_train"] == before["dataset"]["n_train"] + 2


# ----------------------------------------------------------------- 4. metric phân loại hàng loạt
def test_batch_metrics_ignore_rows_without_label():
    ev = evaluate_labels(["spam", "ham", None], ["Spam", "Ham", "Spam"])
    assert ev["n_labeled"] == 2
    assert ev["accuracy"] == 1.0 and ev["precision"] == 1.0 and ev["recall"] == 1.0


def test_batch_metrics_numeric_labels_and_empty():
    ev = evaluate_labels([1, 0, 1, 0], ["Spam", "Spam", "Ham", "Ham"])
    assert ev["precision"] == 0.5 and ev["recall"] == 0.5
    assert evaluate_labels([None, "?"], ["Spam", "Ham"]) is None
    assert evaluate_labels(["ham"], ["Ham"])["precision"] is None  # không có dự đoán spam -> không định nghĩa


# ----------------------------------------------------------------- 5. luật URL
@pytest.mark.parametrize("url, expected", [
    ("https://example.org/docs/bit.ly-guide", False),
    ("https://www.microsoft.com/office", False),   # chứa chuỗi con "t.co"
    ("HTTP://BIT.LY/abc", True),
    ("www.bit.ly/abc).", True),
    ("https://sub.tinyurl.com/x", True),
    ("https://notbit.ly/x", False),
])
def test_shortener_matches_hostname_only(url, expected):
    assert is_shortener(url) is expected


def test_uppercase_scheme_still_triggers_phishing():
    ids = {s["id"] for s in detect_signals("Login HTTPS://evil.example.com and verify your account")}
    assert "phishing" in ids


# ----------------------------------------------------------------- 5. giải thích email dài
def test_explain_marks_tokens_beyond_limit_as_not_analysed(tmp_path):
    model_dir = tmp_path / "m"
    run_training(str(SAMPLE_CSV), model_dir, verbose=False)
    text = " ".join(["free"] * 5 + ["hello"] * 10)
    tokens = explain_email(text, model_dir=model_dir, max_tokens=5)
    assert all(t["impact"] is not None for t in tokens[:5])
    assert all(t["impact"] is None for t in tokens[5:])
    html = ui.highlight_html(tokens)
    assert html.count("Chưa phân tích") == 10 and "\n" not in html


# ----------------------------------------------------------------- 3. diễn giải điểm spam
def test_gauge_does_not_claim_risk_level():
    assert "Rủi ro" not in ui.gauge(90, 0.5)
    assert "Độ tin cậy" not in ui.verdict_card("Spam", 90, "M", 1)
