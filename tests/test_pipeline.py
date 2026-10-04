"""Test cơ bản cho preprocess, train và predict. Chạy: pytest"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import explain_email, predict_batch, predict_email  # noqa: E402
from preprocess import EXTRA_FEATURE_NAMES, clean_text, extra_features  # noqa: E402
from storage import log_feedback, read_feedback  # noqa: E402
from train import MODEL_NAMES, run_training  # noqa: E402

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_spam.csv"
SPAM = "WINNER!! You won a $1000 prize. Click http://bit.ly/claim now to claim your free reward!"
HAM = "Hi team, can we move tomorrow's meeting to 3pm? Please send the report. Thanks"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    run_training(str(SAMPLE_CSV), path, verbose=False)
    return path


# ----------------------------------------------------------------- preprocess
def test_clean_text_replaces_url_and_removes_noise():
    out = clean_text("WINNER!! Visit http://x.com or mail a@b.com for $100")
    assert "urltoken" in out and "emailtoken" in out and "numtoken" in out
    assert "!" not in out and "$" not in out
    assert out == out.lower()


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


def test_extra_features_shape_and_values():
    feats = extra_features([SPAM, HAM, ""])
    assert feats.shape == (3, len(EXTRA_FEATURE_NAMES))
    assert (feats >= 0).all()
    assert feats[0, 0] == 1          # SPAM có 1 link
    assert feats[1, 0] == 0          # HAM không có link
    assert np.isfinite(feats).all()  # chuỗi rỗng không gây chia cho 0


# ----------------------------------------------------------------- train
def test_training_saves_all_models_and_metrics(model_dir):
    for name in MODEL_NAMES:
        assert (model_dir / f"{name}.joblib").exists()
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best"] in MODEL_NAMES
    assert set(metrics["models"]) == set(MODEL_NAMES)
    assert metrics["top_terms"]["spam"] and metrics["top_terms"]["ham"]


# ----------------------------------------------------------------- predict
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_separates_obvious_cases(model_dir, name):
    assert predict_email(SPAM, name, model_dir=model_dir)["label"] == "Spam"
    assert predict_email(HAM, name, model_dir=model_dir)["label"] == "Ham"


def test_predict_output_schema(model_dir):
    res = predict_email(SPAM, model_dir=model_dir)
    assert set(res) == {"label", "confidence", "spam_probability", "model", "threshold"}
    assert 0 <= res["spam_probability"] <= 100 and 50 <= res["confidence"] <= 100


def test_threshold_changes_label(model_dir):
    assert predict_email(SPAM, model_dir=model_dir, threshold=0.01)["label"] == "Spam"
    assert predict_email(SPAM, model_dir=model_dir, threshold=0.999)["label"] == "Ham"


def test_predict_batch_matches_single_and_handles_bad_input(model_dir):
    batch = predict_batch([SPAM, HAM, None], model_dir=model_dir)
    assert [b["label"] for b in batch[:2]] == ["Spam", "Ham"]
    assert len(batch) == 3
    assert predict_batch([], model_dir=model_dir) == []


def test_explain_returns_one_item_per_word(model_dir):
    tokens = explain_email(SPAM, model_dir=model_dir)
    assert [t["token"] for t in tokens] == SPAM.split()
    assert explain_email("", model_dir=model_dir) == []


# ----------------------------------------------------------------- storage
def test_feedback_roundtrip(tmp_path):
    path = tmp_path / "fb.csv"
    log_feedback("hello,\nworld", "nb", "Ham", "Spam", 12.3, path)
    rows = read_feedback(path)
    assert len(rows) == 1 and rows[0]["correct_label"] == "Spam" and rows[0]["text"] == "hello,\nworld"
