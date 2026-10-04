# 📧 Spam Email Detector

Hệ thống phát hiện email/tin nhắn rác (Spam/Ham) bằng Machine Learning: pipeline TF-IDF + đặc trưng thủ công,
**ensemble Soft Voting** của 3 mô hình, kèm giao diện web Streamlit có giải thích kết quả và vòng lặp phản hồi.

<!-- Thêm ảnh chụp màn hình giao diện vào đây, ví dụ: ![Demo](docs/demo.png) -->

## Tính năng

**Mô hình**
- **Ensemble**: MultinomialNB + Logistic Regression + LinearSVC (calibrated) gộp bằng Soft Voting, so sánh với từng mô hình đơn lẻ.
- **Chọn mô hình không rò rỉ dữ liệu**: chọn theo F1 của lớp spam bằng 5-fold cross-validation trên tập train, báo cáo cuối cùng trên tập test.
- **Đặc trưng bổ sung**: số link, số dấu `!`, tỉ lệ chữ IN HOA, tỉ lệ chữ số, độ dài, ký hiệu tiền tệ.
- **Vòng lặp phản hồi**: người dùng bấm 👍/👎 trên giao diện, `python train.py --use-feedback` đưa các phản hồi đó vào lần huấn luyện sau.

**Giao diện (5 trang, menu dọc ở thanh bên)**
- **Tổng quan**: số liệu nhanh của phiên làm việc, cách hệ thống hoạt động.
- **Kiểm tra email**: nhập tiêu đề + nội dung, thẻ kết luận, thanh đo xác suất kèm ngưỡng, **dấu hiệu đáng ngờ theo luật** (link rút gọn, từ khẩn cấp, nhắc đến tiền thưởng, nghi lừa đảo tài khoản...), **đồng thuận giữa các mô hình**, tô màu từng từ đẩy về spam/ham.
- **Phân loại hàng loạt**: tải CSV (có file mẫu), biểu đồ tỉ lệ spam, 10 dòng đáng ngờ nhất, tự tính accuracy/precision/recall nếu file có nhãn, tải kết quả.
- **Hiệu năng mô hình**: so sánh F1/Precision/Recall/Accuracy bằng biểu đồ cột ngang, confusion matrix, từ khóa đặc trưng (tách riêng đặc trưng thủ công).
- **Phản hồi & lịch sử**: lịch sử phiên, thống kê và tải về phản hồi người dùng.
- Điều chỉnh **mô hình** và **ngưỡng phân loại** ngay ở thanh bên; hiển thị tốt ở cả chế độ sáng và tối.

Có unit test (pytest).

## Kiến trúc

```
Văn bản gốc ─┬─ clean_text → TF-IDF (1-2 gram) ─┐
             └─ đặc trưng thủ công → MinMax ────┴─► Soft Voting (NB + LR + SVM) ─► P(spam) ─► Spam/Ham + độ tin cậy
```

Toàn bộ tiền xử lý nằm **bên trong** sklearn `Pipeline`, nên model đã lưu nhận trực tiếp văn bản gốc
(thuận tiện khi bọc thành API).

```
spam_detector/
├── app.py              # UI Streamlit (5 trang)
├── ui_components.py    # Thành phần giao diện HTML/CSS (thuần Python, có test)
├── train.py            # Train 4 model, cross-validation, lưu model + metrics.json
├── predict.py          # predict_email / predict_batch / predict_all_models / explain_email (+ CLI)
├── signals.py          # Phát hiện dấu hiệu đáng ngờ theo luật
├── preprocess.py       # clean_text, đặc trưng thủ công
├── storage.py          # Ghi/đọc feedback (CSV; dễ thay bằng database)
├── tests/              # pytest
├── .streamlit/config.toml
├── data/sample_spam.csv   # 40 mẫu để demo/test; dữ liệu thật tự tải khi train
├── models/             # (tự sinh) *.joblib + metrics.json
└── logs/               # (tự sinh) feedback.csv
```

## Cài đặt & chạy

Yêu cầu Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train.py                   # tải SMS Spam Collection (UCI), train, lưu vào models/
streamlit run app.py              # http://localhost:8501
```

Dùng dòng lệnh:

```bash
python predict.py "WINNER!! Claim your free prize at http://bit.ly/win" --explain
python predict.py --model nb "Hi, can we move the meeting to 3pm?"
```

Dùng dữ liệu riêng: `python train.py --data my_data.csv` (CSV có cột `label` = spam/ham và `text`).
Huấn luyện lại có thêm phản hồi người dùng: `python train.py --use-feedback`.

Chạy test:

```bash
pip install -r requirements-dev.txt
pytest
```

## Kết quả

Chạy `python train.py` để sinh bảng so sánh các mô hình trên tập test (dữ liệu SMS Spam Collection, 5169 mẫu sau khi bỏ trùng),
rồi dán kết quả của bạn vào đây (trang "Hiệu năng mô hình" trong app hiển thị cùng số liệu):

| Model | CV F1 (spam) | Accuracy | Precision | Recall | F1 |
|-------|-------------|----------|-----------|--------|----|
| MultinomialNB | | | | | |
| LogisticRegression | | | | | |
| LinearSVC (calibrated) | | | | | |
| Soft Voting | | | | | |

## Giới hạn

- Chỉ phân loại **nội dung văn bản tiếng Anh**; chưa xử lý file đính kèm, HTML phức tạp, header email (SPF/DKIM).
- Dataset là tin nhắn SMS; với email dài, chất lượng nên được kiểm tra lại bằng dữ liệu email (ví dụ Enron Spam).
- "Dấu hiệu đáng ngờ" là luật đơn giản, chỉ mang tính tham khảo, không ảnh hưởng đến kết luận của mô hình.

## Lộ trình phát triển

- [x] Ensemble nhiều mô hình + chọn theo F1 bằng cross-validation
- [x] UI nhiều trang, giải thích kết quả, phân loại hàng loạt, vòng lặp phản hồi
- [ ] REST API với FastAPI (`POST /predict`, `POST /predict/batch`), UI gọi qua API
- [ ] Dockerfile + GitHub Actions (chạy pytest)
- [ ] Lưu log/feedback vào SQL database (SQLite → PostgreSQL), retrain định kỳ từ feedback
- [ ] Thử mô hình BERT (DistilBERT) để so sánh với ensemble hiện tại
