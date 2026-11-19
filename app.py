"""Giao diện Streamlit cho Spam Email Detector. Chạy: streamlit run app.py"""
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import ui_components as ui
from predict import (DEFAULT_THRESHOLD, evaluate_labels, explain_email, load_metrics, predict_all_models,
                     predict_batch, predict_email)
from email_parser import looks_like_raw_email, parse_eml
from email_risk import assess_risk, attachment_issue, detect_email_signals, link_issues, merge_signals
from signals import detect_signals
from storage import log_feedback, read_feedback

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")
st.markdown(ui.style_tag(), unsafe_allow_html=True)

try:
    metrics = load_metrics()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

MODELS = metrics["models"]
BEST = metrics["best"]
DS = metrics["dataset"]
SAMPLE_CSV = Path(__file__).parent / "data" / "sample_spam.csv"
SAMPLE_EMLS = {p.name: p for p in sorted((Path(__file__).parent / "data" / "samples").glob("*.eml"))}
INPUT_MODES = ["✍️ Dán nội dung", "📎 Tải file .eml", "🧪 Email mẫu"]
RISK_LABELS = {"high": "Cao", "medium": "Thận trọng", "low": "Thấp"}
PAGES = ["🏠 Tổng quan", "🔍 Kiểm tra email", "📂 Phân loại hàng loạt", "📊 Hiệu năng mô hình", "🗂️ Phản hồi & lịch sử"]
EXAMPLES = {
    "— Chọn ví dụ (hoặc để trống) —": ("", ""),
    "Spam: trúng thưởng": ("You have WON a prize!", "WINNER!! You have been selected to receive a $1000 prize. "
                           "Call now or click http://bit.ly/claim to claim your free reward!"),
    "Spam: giả mạo ngân hàng": ("URGENT: account suspended", "Your bank account has been suspended. "
                                "Verify your password at http://secure-login.example.com immediately."),
    "Ham: công việc": ("Meeting moved to 3pm", "Hi team, the meeting is moved to 3pm tomorrow. "
                       "Please send me the updated report before then. Thanks!"),
    "Ham: đời thường": ("Lunch today?", "Hey, are we still on for lunch today? Let me know where you want to meet."),
}
EXTRA_LABELS = {
    "url_count": "Số đường link", "exclamation_count": "Số dấu chấm than", "upper_ratio": "Tỉ lệ chữ IN HOA",
    "digit_ratio": "Tỉ lệ chữ số", "log_length": "Độ dài văn bản", "currency_count": "Ký hiệu tiền tệ ($ £ €)",
}

st.session_state.setdefault("history", [])
st.session_state.setdefault("email_subject", "")
st.session_state.setdefault("email_body", "")


# ----------------------------------------------------------------------------- callbacks
def go_to(page: str) -> None:
    st.session_state["page"] = page


def load_example() -> None:
    subject, body = EXAMPLES[st.session_state["example"]]
    st.session_state["email_subject"], st.session_state["email_body"] = subject, body


def send_feedback(is_correct: bool) -> None:
    last = st.session_state.get("last")
    if not last or last.get("feedback"):
        return
    predicted = last["result"]["label"]
    correct = predicted if is_correct else ("Ham" if predicted == "Spam" else "Spam")
    log_feedback(last["text"], last["result"]["model"], predicted, correct, last["result"]["spam_probability"])
    last["feedback"] = correct


def clear_history() -> None:
    st.session_state["history"] = []


def clear_batch() -> None:
    """Đổi/bỏ file CSV thì xóa kết quả cũ, tránh hiển thị kết quả của file trước."""
    st.session_state.pop("batch", None)


# ----------------------------------------------------------------------------- helpers
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📧 Spam Detector")
    page = st.radio("Trang", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**⚙️ Cài đặt**")
    model_name = st.selectbox(
        "Mô hình", list(MODELS), index=list(MODELS).index(BEST),
        format_func=lambda n: MODELS[n]["label"] + (" ⭐" if n == BEST else ""),
    )
    threshold = st.slider(
        "Ngưỡng phân loại spam", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05,
        help="Email bị coi là spam khi xác suất spam ≥ ngưỡng. Tăng ngưỡng: ít báo nhầm email thường hơn "
             "nhưng dễ bỏ sót spam hơn.",
    )
    st.divider()
    st.caption(f"Dữ liệu: {DS['source']}")
    st.caption(f"{DS['n_samples']:,} mẫu ({DS['n_spam']:,} spam / {DS['n_ham']:,} ham)")
    st.caption(f"Huấn luyện lúc: {metrics['created_at']}")

history = st.session_state["history"]

# ============================================================================= 1. TỔNG QUAN
if page == PAGES[0]:
    html(ui.hero("Spam Email Detector",
                 "Kiểm tra email đáng ngờ: phát hiện giả danh thương hiệu, tên miền nhái, link lệch địa chỉ, "
                 "file đính kèm nguy hiểm, kèm mô hình Machine Learning phân loại spam có giải thích.",
                 ["Đọc file .eml", "Dấu hiệu lừa đảo tiếng Việt + Anh", "Soft Voting ensemble"]))
    st.button("🔍 Thử ngay", type="primary", on_click=go_to, args=(PAGES[1],))

    st.subheader("Số liệu nhanh")
    html(ui.kpi_list([
        ("Đã kiểm tra trong phiên này", len(history)),
        ("Trong đó rủi ro cao", sum(1 for h in history if h.get("Rủi ro") == RISK_LABELS["high"])),
        ("Mô hình ML cho là spam", sum(1 for h in history if h["Kết quả"] == "Spam")),
        ("Phản hồi người dùng đã gửi", len(read_feedback())),
        (f"F1 spam của mô hình tốt nhất ({MODELS[BEST]['label']})", f"{MODELS[BEST]['f1']:.3f}"),
        ("Số mẫu huấn luyện", f"{DS['n_samples']:,}"),
    ]))

    st.subheader("Cách hoạt động")
    html(ui.card("Từ email đến kết luận", ui.step_list([
        ("Đưa email vào", "Dán nội dung, dán nguồn email hoặc tải file .eml ở trang Kiểm tra email."),
        ("Kiểm tra dấu hiệu", "Người gửi giả danh, tên miền nhái, SPF/DKIM/DMARC, link lệch địa chỉ, file đính kèm "
                              "nguy hiểm, từ khóa lừa đảo tiếng Việt và tiếng Anh."),
        ("Làm sạch và trích đặc trưng", "Chuẩn hóa văn bản, TF-IDF (1-2 từ) cùng số link, chữ HOA, dấu !, ký hiệu tiền..."),
        ("Nhiều mô hình bỏ phiếu", "Naive Bayes, Logistic Regression và SVM cho xác suất, Soft Voting lấy trung bình."),
        ("Kết luận kèm giải thích", "Spam hay Ham, điểm spam của mô hình, từ nào ảnh hưởng nhất và các dấu hiệu đáng ngờ."),
    ])))
    st.caption("Giới hạn: mô hình ML hiện học từ SMS tiếng Anh; dấu hiệu theo luật không bắt được mọi kiểu lừa đảo; "
               "file đính kèm chỉ được kiểm tra theo tên, không quét virus.")

# ============================================================================= 2. KIỂM TRA EMAIL
elif page == PAGES[1]:
    st.header("🔍 Kiểm tra email")
    mode = st.radio("Nguồn email", INPUT_MODES, horizontal=True, key="input_mode", label_visibility="collapsed")
    eml_file, sample_name = None, None
    if mode == INPUT_MODES[0]:
        st.selectbox("Ví dụ nhanh", list(EXAMPLES), key="example", on_change=load_example)
        st.text_input("Tiêu đề (tùy chọn)", key="email_subject", placeholder="Ví dụ: You have WON a prize!")
        st.text_area("Nội dung email / tin nhắn", key="email_body", height=170,
                     placeholder="Dán nội dung email vào đây, hoặc dán cả NGUỒN email để phân tích người gửi và link...")
        st.caption("Mẹo: dán **nguồn email** (Gmail: ⋮ → *Hiển thị bản gốc*; Outlook: … → *Xem* → *Xem nguồn thư*) "
                   "để kiểm tra cả người gửi, SPF/DKIM/DMARC và địa chỉ thật của các link.")
    elif mode == INPUT_MODES[1]:
        eml_file = st.file_uploader("File .eml", type=["eml"])
        st.caption("Lấy file .eml: Gmail → mở thư → ⋮ → *Tải thư xuống*. Outlook → mở thư → … → *Lưu dưới dạng* "
                   "(hoặc kéo thư ra thư mục). File chỉ được đọc trong phiên này, không lưu lại.")
    else:
        sample_name = st.selectbox("Email mẫu", list(SAMPLE_EMLS)) if SAMPLE_EMLS else None
        st.caption("Email mẫu dùng tên miền thử nghiệm (.test / .example), không trỏ đến trang thật.")

    if st.button("Phân tích", type="primary"):
        parsed, error = None, None
        try:
            if mode == INPUT_MODES[1]:
                if eml_file is None:
                    error = "Vui lòng chọn file .eml."
                else:
                    parsed = parse_eml(eml_file.getvalue())
            elif mode == INPUT_MODES[2]:
                if sample_name:
                    parsed = parse_eml(SAMPLE_EMLS[sample_name].read_bytes())
                else:
                    error = "Không tìm thấy email mẫu trong data/samples/."
            elif looks_like_raw_email(st.session_state["email_body"]):
                parsed = parse_eml(st.session_state["email_body"])
        except ValueError as exc:
            error = str(exc)

        full_text = parsed.model_text() if parsed else (
            st.session_state["email_subject"].strip() + "\n" + st.session_state["email_body"].strip()).strip()
        if error:
            st.warning(error)
        elif not full_text:
            st.warning("Vui lòng nhập nội dung (email không có nội dung văn bản để phân tích).")
        else:
            start = time.perf_counter()
            result = predict_email(full_text, model_name, threshold)
            tokens = explain_email(full_text, model_name)
            consensus = predict_all_models(full_text, threshold)
            signals = merge_signals(detect_signals(full_text), detect_email_signals(parsed))
            risk = assess_risk(signals)
            elapsed = time.perf_counter() - start
            st.session_state["last"] = {"text": full_text, "result": result, "tokens": tokens, "signals": signals,
                                        "risk": risk, "email": parsed, "consensus": consensus, "elapsed": elapsed,
                                        "feedback": None}
            preview = (f"{parsed.from_addr} · {parsed.subject}" if parsed else full_text).replace("\n", " ")
            history.insert(0, {
                "Thời gian": datetime.now().strftime("%H:%M:%S"),
                "Nội dung": preview[:70] + ("…" if len(preview) > 70 else ""),
                "Rủi ro": RISK_LABELS[risk["level"]],
                "Kết quả": result["label"],
                "Xác suất spam (%)": result["spam_probability"],
                "Mô hình": MODELS[model_name]["label"],
            })

    last = st.session_state.get("last")
    if last:
        res = last["result"]
        if res["model"] != model_name or abs(res["threshold"] - threshold) > 1e-9:
            st.info("Bạn vừa đổi mô hình hoặc ngưỡng ở thanh bên. Bấm **Phân tích** để cập nhật kết quả.")

        # 1) Rủi ro theo dấu hiệu (người gửi, xác thực, link, file đính kèm, từ khóa) - phần chính
        html(ui.risk_card(last.get("risk") or assess_risk(last["signals"])))
        parsed = last.get("email")
        if parsed:
            html(ui.email_header_card(parsed))
        html(ui.signals_html(last["signals"]))
        if parsed:
            html(ui.links_html(parsed.links, link_issues))
            html(ui.attachments_html(parsed.attachments, attachment_issue))

        # 2) Mô hình ML - tham khảo
        st.subheader("Đánh giá của mô hình ML")
        st.caption("Mô hình hiện được huấn luyện trên tin nhắn SMS tiếng Anh, nên với email (nhất là tiếng Việt) "
                   "kết quả này chỉ mang tính tham khảo. Mức rủi ro phía trên dựa trên dấu hiệu cụ thể.")
        html(ui.verdict_card(res["label"], res["confidence"], MODELS[res["model"]]["label"], last["elapsed"] * 1000))
        html(ui.gauge(res["spam_probability"], res["threshold"]))

        votes = sum(1 for r in last["consensus"] if r["label"] == "Spam")
        html(ui.card(
            f"Đồng thuận giữa các mô hình: {votes}/{len(last['consensus'])} cho là Spam",
            ui.hbar_chart([(r["model_label"], r["spam_probability"],
                            ui.SPAM_COLOR if r["label"] == "Spam" else ui.HAM_COLOR) for r in last["consensus"]])
            + '<div class="sd-note">Độ dài thanh = xác suất spam. Đỏ: kết luận Spam, xanh: kết luận Ham.</div>',
        ))

        with st.expander("Vì sao mô hình ML cho kết quả này?"):
            html(ui.highlight_html(last["tokens"]))
            st.caption("🟥 từ đẩy về phía spam · 🟩 từ đẩy về phía ham · rê chuột lên từ để xem mức ảnh hưởng")
            n_skipped = sum(1 for t in last["tokens"] if t["impact"] is None)
            if n_skipped:
                st.caption(f"⚪ {n_skipped} từ cuối (gạch chân chấm) **chưa được phân tích** để giữ tốc độ, "
                           "không có nghĩa là chúng không ảnh hưởng đến kết quả.")

        st.divider()
        if last["feedback"]:
            st.success(f"Cảm ơn! Đã ghi nhận nhãn đúng là **{last['feedback']}** vào logs/feedback.csv")
        else:
            st.write("Kết luận Spam/Ham của mô hình có đúng không?")
            c1, c2, _ = st.columns([1, 1, 4])
            c1.button("👍 Đúng", on_click=send_feedback, args=(True,))
            c2.button("👎 Sai", on_click=send_feedback, args=(False,))

# ============================================================================= 3. HÀNG LOẠT
elif page == PAGES[2]:
    st.header("📂 Phân loại hàng loạt")
    st.write("Tải lên file CSV có một cột chứa nội dung email/tin nhắn để phân loại cả file cùng lúc.")
    if SAMPLE_CSV.exists():
        st.download_button("⬇️ Tải file CSV mẫu để thử", SAMPLE_CSV.read_bytes(), "sample_spam.csv", "text/csv")
    file = st.file_uploader("File CSV", type=["csv"], on_change=clear_batch)

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không đọc được file CSV: {exc}")
            df = None
        if df is not None and not df.empty:
            text_col = st.selectbox("Cột nội dung", list(df.columns), index=ui.guess_text_column(df))
            label_col = st.selectbox("Cột nhãn thật (tùy chọn, để đánh giá)", ["(không có)"] + list(df.columns))
            st.caption("Xem trước 5 dòng đầu:")
            st.dataframe(df.head(5))

            if st.button("Phân loại tất cả", type="primary"):
                with st.spinner(f"Đang phân loại {len(df):,} dòng..."):
                    preds = predict_batch(df[text_col].tolist(), model_name, threshold)
                out = df.copy()
                out["predicted_label"] = [p["label"] for p in preds]
                out["spam_probability_pct"] = [p["spam_probability"] for p in preds]
                out["confidence_pct"] = [p["confidence"] for p in preds]
                st.session_state["batch"] = {"df": out, "text_col": text_col,
                                             "label_col": None if label_col == "(không có)" else label_col}

    batch = st.session_state.get("batch")
    if batch:
        out, text_col = batch["df"], batch["text_col"]
        n_spam = int((out["predicted_label"] == "Spam").sum())
        st.subheader("Kết quả")
        html(ui.card("Tỉ lệ spam trong file", ui.donut(n_spam, len(out) - n_spam)))

        if batch["label_col"] and batch["label_col"] in out.columns:
            ev = evaluate_labels(out[batch["label_col"]], out["predicted_label"])
            if ev:
                def pct(v):
                    return "—" if v is None else f"{v:.1%}"
                html(ui.kpi_list([
                    ("Accuracy", pct(ev["accuracy"])),
                    ("Precision (spam)", pct(ev["precision"])),
                    ("Recall (spam)", pct(ev["recall"])),
                ]))
                if ev["n_labeled"] < len(out):
                    st.caption(f"Chỉ số tính trên {ev['n_labeled']:,}/{len(out):,} dòng có nhãn hợp lệ.")
            else:
                st.warning("Cột nhãn không có giá trị hợp lệ (spam/ham hoặc 1/0) nên không tính được độ chính xác.")

        st.subheader("10 dòng đáng ngờ nhất")
        top = out.sort_values("spam_probability_pct", ascending=False).head(10)
        st.dataframe(pd.DataFrame({
            "Nội dung": top[text_col].astype(str).str.slice(0, 90),
            "Kết quả": top["predicted_label"],
            "Xác suất spam (%)": top["spam_probability_pct"],
        }), hide_index=True)

        with st.expander("Xem toàn bộ kết quả"):
            st.dataframe(out, height=360)
        st.download_button("⬇️ Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           "spam_predictions.csv", "text/csv")

# ============================================================================= 4. HIỆU NĂNG
elif page == PAGES[3]:
    st.header("📊 Hiệu năng mô hình")
    st.subheader("Dữ liệu huấn luyện")
    kpis = [("Tổng số mẫu", f"{DS['n_samples']:,}"), ("Spam", f"{DS['n_spam']:,}"), ("Ham", f"{DS['n_ham']:,}"),
            ("Tập train / test", f"{DS['n_train']:,} / {DS['n_test']:,}"),
            ("Cross-validation", f"{DS['cv_folds']} folds" if DS["cv_folds"] >= 2 else "không")]
    if DS.get("n_feedback"):
        kpis.append(("Phản hồi người dùng đã đưa vào train", f"{DS.get('n_feedback_used', DS['n_feedback']):,}"))
        if DS.get("n_feedback_in_test"):
            kpis.append(("Phản hồi bị bỏ qua (trùng tập test)", f"{DS['n_feedback_in_test']:,}"))
    html(ui.kpi_list(kpis))

    st.subheader("So sánh các mô hình (tập test)")
    st.caption(f"⭐ {MODELS[BEST]['label']} được chọn theo tiêu chí `{metrics['selection_metric']}`. "
               "Thanh dài đủ 100% nghĩa là điểm tuyệt đối.")
    for title, key, hint in [
        ("F1 (spam)", "f1", "cân bằng giữa bắt được spam và không báo nhầm"),
        ("Precision (spam)", "precision", "trong số email bị báo spam, bao nhiêu thật sự là spam"),
        ("Recall (spam)", "recall", "trong số spam thật, bắt được bao nhiêu"),
        ("Accuracy", "accuracy", "tỉ lệ dự đoán đúng chung"),
    ]:
        ranked = sorted(MODELS, key=lambda n: MODELS[n][key], reverse=True)
        html(ui.card(f"{title} — {hint}", ui.hbar_chart(
            [(MODELS[n]["label"] + (" ⭐" if n == BEST else ""), MODELS[n][key],
              ui.ACCENT if n == BEST else "#94a3b8") for n in ranked], max_value=1.0, fmt="{:.1%}")))

    with st.expander("Bảng số liệu chi tiết"):
        st.dataframe(pd.DataFrame([
            {"Model": m["label"], "CV F1 (spam)": m["cv_f1_mean"], "Accuracy": m["accuracy"],
             "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]} for m in MODELS.values()
        ]), hide_index=True)

    st.subheader(f"Confusion matrix — {MODELS[model_name]['label']}")
    html(ui.card("Mỗi ô là số email; hàng là nhãn thật, cột là dự đoán", ui.confusion_matrix_html(
        MODELS[model_name]["confusion_matrix"])))
    st.caption("Đổi mô hình ở thanh bên để xem confusion matrix của mô hình khác.")

    st.subheader("Từ khóa đặc trưng (theo Logistic Regression)")
    terms = metrics["top_terms"]
    html(ui.card("Từ gợi ý SPAM mạnh nhất", ui.hbar_chart(
        [(t["term"], t["weight"]) for t in terms["spam"]], fmt="{:.2f}", default_color=ui.SPAM_COLOR)))
    html(ui.card("Từ gợi ý HAM mạnh nhất", ui.hbar_chart(
        [(t["term"], abs(t["weight"])) for t in terms["ham"]], fmt="{:.2f}", default_color=ui.HAM_COLOR)))
    if terms.get("extra"):
        html(ui.card("Đặc trưng thủ công (đỏ: đẩy về spam, xanh: đẩy về ham)", ui.hbar_chart(
            [(EXTRA_LABELS.get(t["term"], t["term"]), t["weight"],
              ui.SPAM_COLOR if t["weight"] > 0 else ui.HAM_COLOR) for t in terms["extra"]], fmt="{:+.2f}")))
        st.caption("Thang đo của các đặc trưng thủ công khác với từ, nên được tách riêng, không so sánh trực tiếp với từ khóa.")

# ============================================================================= 5. PHẢN HỒI & LỊCH SỬ
else:
    st.header("🗂️ Phản hồi & lịch sử")

    st.subheader("Lịch sử trong phiên làm việc")
    if history:
        hist = pd.DataFrame(history)
        st.dataframe(hist, hide_index=True)
        st.download_button("⬇️ Tải lịch sử (CSV)", hist.to_csv(index=False).encode("utf-8-sig"), "history.csv", "text/csv")
        st.button("Xóa lịch sử", on_click=clear_history)
    else:
        st.info("Chưa có lần kiểm tra nào. Hãy sang trang Kiểm tra email.")

    st.subheader("Phản hồi của người dùng")
    rows = read_feedback()
    if rows:
        fb = pd.DataFrame(rows)
        agree = int((fb["predicted"] == fb["correct_label"]).sum())
        html(ui.kpi_list([
            ("Tổng số phản hồi", len(fb)),
            ("Mô hình dự đoán đúng", f"{agree} ({agree / len(fb):.0%})"),
            ("Mô hình dự đoán sai", len(fb) - agree),
        ]))
        st.dataframe(fb.drop(columns=["spam_probability"], errors="ignore"), hide_index=True)
        st.download_button("⬇️ Tải phản hồi (CSV)", fb.to_csv(index=False).encode("utf-8-sig"), "feedback.csv", "text/csv")
        st.markdown("**Dùng phản hồi để huấn luyện lại mô hình** (chạy trong terminal, sau đó tải lại trang):")
        st.code("python train.py --use-feedback", language="bash")
    else:
        st.info("Chưa có phản hồi nào. Sau khi phân loại, bấm 👍/👎 ở trang Kiểm tra email để góp dữ liệu cải thiện mô hình.")
