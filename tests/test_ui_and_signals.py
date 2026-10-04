"""Test cho signals, ui_components, predict_all_models và huấn luyện có phản hồi. Chạy: pytest"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui_components as ui  # noqa: E402
from predict import predict_all_models  # noqa: E402
from preprocess import EXTRA_FEATURE_NAMES, clean_text  # noqa: E402
from signals import detect_signals  # noqa: E402
from storage import log_feedback  # noqa: E402
from train import MODEL_NAMES, run_training  # noqa: E402

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_spam.csv"
SPAM = "WINNER!! You won a $1000 prize. Click http://bit.ly/claim now to claim your free reward!"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    run_training(str(SAMPLE_CSV), path, verbose=False)
    return path


# ----------------------------------------------------------------- signals
def test_signals_detect_shortened_link_and_money():
    ids = {s["id"] for s in detect_signals(SPAM)}
    assert {"short_link", "money"} <= ids


def test_signals_phishing_needs_credentials_plus_link():
    text = "URGENT: your bank account is suspended. Verify your password at http://secure.example.com"
    assert "phishing" in {s["id"] for s in detect_signals(text)}
    assert "phishing" not in {s["id"] for s in detect_signals("Please verify your account details with the bank")}


def test_signals_clean_email_has_none_and_bad_input_is_safe():
    assert detect_signals("Hi team, the meeting is at 3pm tomorrow. Thanks") == []
    assert detect_signals("") == [] and detect_signals(None) == []


def test_signals_sorted_most_severe_first():
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[s["severity"]] for s in detect_signals(SPAM + " !!!")]
    assert ranks == sorted(ranks)


# ----------------------------------------------------------------- ui_components
def test_ui_components_return_single_line_html():
    parts = [
        ui.style_tag(), ui.hero("T", "S", ["a", "b"]), ui.kpi_list([("a", 1), ("b", "2")]),
        ui.verdict_card("Spam", 97.5, "Model", 12), ui.verdict_card("Ham", 80, "Model", 5),
        ui.gauge(50, 0.5), ui.signals_html([]), ui.signals_html(detect_signals(SPAM)),
        ui.hbar_chart([("a", 1), ("b", 2, "#fff")]), ui.donut(1, 2), ui.donut(0, 0),
        ui.confusion_matrix_html({"tn": 5, "fp": 1, "fn": 2, "tp": 3}), ui.step_list([("t", "d")]),
        ui.highlight_html([{"token": "x", "impact": 0.3}, {"token": "y", "impact": 0.0}]), ui.card("t", "<b>x</b>"),
    ]
    for part in parts:
        assert "\n" not in part  # Markdown sẽ coi HTML nhiều dòng/thụt đầu dòng là code block


def test_ui_components_escape_user_text():
    out = ui.highlight_html([{"token": "<script>alert(1)</script>", "impact": 0.5}])
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>x</b>" not in ui.hero("<b>x</b>", "s")


def test_hbar_chart_scaling_and_empty():
    assert "width:100.0%" in ui.hbar_chart([("a", 5)])
    assert ui.hbar_chart([]) == ""
    assert "width:50.0%" in ui.hbar_chart([("a", 1), ("b", 2)])


def test_confusion_matrix_and_donut_handle_zero_rows():
    assert "—" in ui.confusion_matrix_html({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    assert "0%" in ui.donut(0, 0)


# ----------------------------------------------------------------- predict / train
def test_predict_all_models_covers_every_model(model_dir):
    results = predict_all_models(SPAM, model_dir=model_dir)
    assert {r["model"] for r in results} == set(MODEL_NAMES)
    assert all("model_label" in r for r in results)


def test_top_terms_keep_extra_features_separate(model_dir):
    terms = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["top_terms"]
    assert {t["term"] for t in terms["extra"]} == set(EXTRA_FEATURE_NAMES)
    words = {t["term"] for t in terms["spam"] + terms["ham"]}
    assert not words & set(EXTRA_FEATURE_NAMES)


def test_training_can_include_user_feedback(tmp_path):
    fb = tmp_path / "fb.csv"
    log_feedback("brand new crypto giveaway click here to claim now", "nb", "Ham", "Spam", 10.0, fb)
    log_feedback("see you at the standup tomorrow morning", "nb", "Spam", "Ham", 90.0, fb)
    metrics = run_training(str(SAMPLE_CSV), tmp_path / "m", use_feedback=True, feedback_path=fb, verbose=False)
    assert metrics["dataset"]["n_feedback"] == 2
    assert metrics["dataset"]["n_samples"] == 42


def test_clean_text_unescapes_html_entities():
    tokens = clean_text("&lt;#&gt; call me &amp; come").split()
    assert "lt" not in tokens and "gt" not in tokens and "call" in tokens
