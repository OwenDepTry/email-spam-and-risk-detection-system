# 📧 Spam Email Detector

[![Tests](https://github.com/OwenDepTry/email-spam-and-risk-detection-system/actions/workflows/tests.yml/badge.svg)](https://github.com/OwenDepTry/email-spam-and-risk-detection-system/actions/workflows/tests.yml)

Công cụ kiểm tra email đáng ngờ, tập trung vào **lừa đảo (phishing)**, hỗ trợ tiếng Việt và tiếng Anh:
dán nội dung / nguồn email hoặc tải file `.eml`, hệ thống cho biết **mức rủi ro và lý do cụ thể**
(giả danh thương hiệu, tên miền nhái, SPF/DKIM/DMARC, link lệch địa chỉ, file đính kèm nguy hiểm...).
Kèm mô hình Machine Learning phân loại spam (TF-IDF + **ensemble Soft Voting** của 3 mô hình) có giải thích kết quả
và vòng lặp phản hồi.

<!-- Thêm ảnh chụp màn hình giao diện vào đây, ví dụ: ![Demo](docs/demo.png) -->

## Tính năng

**Kiểm tra email lừa đảo (theo luật, có giải thích)** — `email_parser.py`, `email_risk.py`
- **Đọc email thật**: file `.eml` hoặc nguồn email dán vào (tự nhận ra); tách người gửi, Reply-To, tiêu đề,
  nội dung (HTML → văn bản), link (chữ hiển thị + địa chỉ thật) và file đính kèm. Chỉ dùng thư viện chuẩn.
- **Người gửi**: tên hiển thị giả danh thương hiệu (ngân hàng, ví điện tử, sàn TMĐT, Tổng cục Thuế, PayPal,
  Microsoft...) nhưng gửi từ tên miền khác / Gmail; tên miền nhái (`vietcombamk`, `paypa1`, `rnicrosoft`,
  ký tự Cyrillic/punycode); Reply-To trỏ đi nơi khác.
- **Xác thực**: đọc SPF / DKIM / DMARC từ header `Authentication-Results` do máy chủ nhận ghi (bỏ qua header giả chèn bên dưới).
- **Link**: chữ hiển thị là một địa chỉ nhưng mở ra địa chỉ khác, link đến tên miền nhái, link thẳng địa chỉ IP, link rút gọn.
- **File đính kèm**: đuôi kép (`hoa_don.pdf.exe`), file chạy được, file `.html` (trang đăng nhập giả), Office có macro, file nén.
- **Từ khóa tiếng Việt + tiếng Anh**: khẩn cấp, trúng thưởng, xác minh / OTP / mật khẩu... (bắt cả khi viết không dấu).
- **Mức rủi ro tổng hợp** (Cao / Thận trọng / Thấp) kèm danh sách lý do; 3 email mẫu trong `data/samples/`.
- Tránh báo nhầm: không so độ giống với tên ngắn (abbank/mbbank, lpbank/tpbank là ngân hàng thật khác nhau),
  không coi "tài khoản"/"ngân hàng" + link là lừa đảo (thông báo số dư thật vẫn ở mức thấp).

**Mô hình**
- **Ensemble**: MultinomialNB + Logistic Regression + LinearSVC (calibrated) gộp bằng Soft Voting, so sánh với từng mô hình đơn lẻ.
- **Chọn mô hình không rò rỉ dữ liệu**: chọn theo F1 của lớp spam bằng 5-fold cross-validation trên tập train, báo cáo cuối cùng trên tập test.
- **Đặc trưng bổ sung**: số link, số dấu `!`, tỉ lệ chữ IN HOA, tỉ lệ chữ số, độ dài, ký hiệu tiền tệ.
- **Vòng lặp phản hồi**: người dùng bấm 👍/👎 trên giao diện, `python train.py --use-feedback` đưa các phản hồi đó vào lần huấn luyện sau.

**Giao diện (5 trang, menu dọc ở thanh bên)**
- **Tổng quan**: số liệu nhanh của phiên làm việc, cách hệ thống hoạt động.
- **Kiểm tra email**: dán nội dung, dán nguồn email, tải file `.eml` hoặc chọn email mẫu → thẻ **mức rủi ro**,
  thông tin người gửi + SPF/DKIM/DMARC, danh sách dấu hiệu, bảng link (chữ hiển thị ↔ trang thật sự mở),
  file đính kèm; bên dưới là kết luận của mô hình ML, thanh đo xác suất, **đồng thuận giữa các mô hình**, tô màu từng từ.
- **Phân loại hàng loạt**: tải CSV (có file mẫu), biểu đồ tỉ lệ spam, 10 dòng đáng ngờ nhất, tự tính accuracy/precision/recall nếu file có nhãn, tải kết quả.
- **Hiệu năng mô hình**: so sánh F1/Precision/Recall/Accuracy bằng biểu đồ cột ngang, confusion matrix, từ khóa đặc trưng (tách riêng đặc trưng thủ công).
- **Phản hồi & lịch sử**: lịch sử phiên, thống kê và tải về phản hồi người dùng.
- Điều chỉnh **mô hình** và **ngưỡng phân loại** ngay ở thanh bên; hiển thị tốt ở cả chế độ sáng và tối.

Có unit test (pytest).

## Kiến trúc

```
Văn bản gốc ─┬─ clean_text → TF-IDF (1-2 gram) ─┐
             └─ đặc trưng thủ công → MinMax ────┴─► Soft Voting (NB + LR + SVM) ─► P(spam) ─► Spam/Ham + điểm spam
```

Toàn bộ tiền xử lý nằm **bên trong** sklearn `Pipeline`, nên model đã lưu nhận trực tiếp văn bản gốc
(thuận tiện khi bọc thành API).

```
spam_detector/
├── app.py              # UI Streamlit (5 trang)
├── ui_components.py    # Thành phần giao diện HTML/CSS (thuần Python, có test)
├── train.py            # Train 4 model, cross-validation, lưu model + metrics.json
├── predict.py          # predict_email / predict_batch / predict_all_models / explain_email (+ CLI)
├── signals.py          # Dấu hiệu đáng ngờ trong nội dung chữ (tiếng Anh + tiếng Việt)
├── email_parser.py     # Đọc file .eml / nguồn email: header, HTML → văn bản, link, file đính kèm
├── email_risk.py       # Dấu hiệu từ cấu trúc email (người gửi, xác thực, link, đính kèm) + mức rủi ro
├── preprocess.py       # clean_text, đặc trưng thủ công
├── storage.py          # Ghi/đọc feedback (CSV; dễ thay bằng database)
├── notebooks/          # Jupyter: khám phá dữ liệu & thử nghiệm (import từ các file .py)
├── tests/              # pytest
├── .streamlit/config.toml
├── data/sample_spam.csv   # 40 mẫu để demo/test; dữ liệu thật tự tải khi train
├── data/samples/*.eml     # 3 email mẫu (lừa đảo VI, lừa đảo EN, hợp lệ) - chỉ dùng tên miền .test/.example
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
python email_risk.py data/samples/*.eml   # kiểm tra dấu hiệu lừa đảo của file .eml (không cần train)
```

Dùng dữ liệu riêng: `python train.py --data my_data.csv` (CSV có cột `label` = spam/ham và `text`).
Huấn luyện lại có thêm phản hồi người dùng: `python train.py --use-feedback`.
Phản hồi chỉ được đưa vào tập **train**; tập test được khóa trong `models/test_split.json` nên các lần
retrain so sánh được với nhau. Phản hồi trùng với tin trong tập test bị bỏ qua; phản hồi cho một tin đã có
trong train sẽ thay nhãn gốc (phản hồi mới nhất được ưu tiên). Muốn chia lại tập test: `python train.py --resplit`.

Chạy test:

```bash
pip install -r requirements-dev.txt
pytest
```

## Notebook (thử nghiệm)

Code chính nằm trong các file `.py`. Notebook trong `notebooks/` chỉ **import** từ đó để khám phá dữ liệu,
thử tham số và soi các tin bị đoán sai; ý tưởng nào tốt thì mới chuyển vào `.py` và thêm test.

```bash
pip install -r requirements-dev.txt
nbstripout --install      # chạy 1 lần sau khi clone: tự xóa output notebook khi commit (diff gọn)
jupyter lab notebooks/01_eda_and_experiments.ipynb   # hoặc mở bằng VS Code (extension Jupyter)
```

`notebooks/01_eda_and_experiments.ipynb`: phân bố độ dài và đặc trưng thủ công theo nhãn, ví dụ `clean_text`,
thử `min_df` của TF-IDF bằng cross-validation, phân tích lỗi trên tập test, precision/recall theo ngưỡng.
Notebook chia dữ liệu với `split_file=None` nên không đụng tới tập test đã khóa trong `models/`.

## Kết quả

SMS Spam Collection (UCI), 5.574 tin gốc, còn 5.160 mẫu sau khi bỏ trùng (642 spam / 4.518 ham). Chia **theo nhóm**:
4.128 train / 1.032 test (128 spam), seed 42, 5-fold `StratifiedGroupKFold` trên train,
scikit-learn 1.9.1. Precision / Recall / F1 tính cho lớp spam.

| Model | CV F1 (spam) | Accuracy | Precision | Recall | F1 | FP | FN |
|-------|-------------|----------|-----------|--------|----|----|----|
| MultinomialNB | 94,85% | 98,74% | 98,32% | 91,41% | 94,74% | 2 | 11 |
| LogisticRegression | 89,60% | 98,16% | 90,37% | 95,31% | 92,78% | 13 | 6 |
| LinearSVC (calibrated) | 94,41% | 98,93% | 98,35% | 92,97% | 95,58% | 2 | 9 |
| Soft Voting ⭐ | 94,98% | 99,13% | 98,37% | 94,53% | 96,41% | 2 | 7 |

⭐ được chọn theo CV F1 trên train. Trung bình 5 lần chia (seed 0–4):

| Model | CV F1 (spam) | F1 trên test |
|-------|-------------|--------------|
| MultinomialNB | 95,08% ± 0,23 | 94,36% ± 0,99 |
| LinearSVC (calibrated) | 94,46% ± 0,23 | 94,17% ± 1,25 |
| Soft Voting | 95,10% ± 0,24 | 95,06% ± 1,05 |

NB và Soft Voting ngang nhau theo CV; trên test Soft Voting nhỉnh hơn ~0,7 điểm (tốt hơn ở 4/5 lần chia),
nên lựa chọn giữa hai mô hình này nằm trong mức nhiễu.

> Trước 19/11/2026, `train.py` đọc file UCI với xử lý dấu ngoặc kép mặc định của pandas: 3 tin bị gộp thành 1
> (5.572 thay vì 5.574 dòng) và khoảng 50 tin bị bỏ dấu `"` ở đầu/cuối. Các con số trên đã tính lại với cách đọc đúng
> (dữ liệu đổi nên tập test cũng được chia lại). Nếu máy bạn còn `data/sms_spam.csv` cũ, hãy xóa file đó để tải lại.

**Về cách chia dữ liệu.** Phiên bản trước chia ngẫu nhiên sau khi chỉ bỏ trùng theo chuỗi gốc,
nên 33/1.034 tin trong test trùng với tin trong train sau làm sạch. Kiểm tra trên 5 seed (F1 spam trên test,
tính trên dữ liệu đọc theo cách cũ, 5.158 mẫu):

| Cách chia | MultinomialNB | Soft Voting |
|-----------|---------------|-------------|
| Ngẫu nhiên (cũ) | 95,99% ± 1,43 | 95,48% ± 1,40 |
| Theo nhóm (mới) | 95,13% ± 0,27 | 95,81% ± 0,55 |

Rò rỉ làm kết quả NB lạc quan khoảng 1 điểm; với Soft Voting không thấy khác biệt rõ.
Kết quả theo một seed có thể dao động hơn 1 điểm, nên khi so sánh mô hình nên báo cáo trung bình nhiều lần chia.

**Lưu ý:** đây là dữ liệu SMS. Chưa thể dùng các con số này để kết luận về email hay phishing thực tế.

## Giới hạn

- **Mô hình ML hiện học từ SMS tiếng Anh** (bước làm sạch còn bỏ chữ có dấu), nên với email — nhất là tiếng Việt —
  kết luận Spam/Ham của mô hình chỉ để tham khảo (ví dụ email lịch thi hợp lệ trong `data/samples/` bị chấm ~98% spam).
  Mức rủi ro trên giao diện dựa trên dấu hiệu theo luật, độc lập với mô hình.
- Dấu hiệu theo luật không bắt được mọi kiểu lừa đảo (ví dụ email từ tài khoản thật bị chiếm quyền, nội dung "sạch").
  Danh sách thương hiệu / tên miền chính thức trong `email_risk.BRANDS` cần được bổ sung dần; ngưỡng mức rủi ro
  chưa được hiệu chỉnh bằng dữ liệu thật.
- SPF/DKIM/DMARC chỉ có khi tải file `.eml` / dán nguồn email; file đính kèm chỉ được kiểm tra theo tên, không quét virus.
- Nội dung email chỉ được lưu khi người dùng bấm 👍/👎 (vào `logs/feedback.csv`); trang phản hồi chưa có đăng nhập.

## Lộ trình phát triển

**Giai đoạn 1 — phân tích rủi ro không cần ML**
- [x] Đọc `.eml` / nguồn email; dấu hiệu người gửi, xác thực, link, file đính kèm; từ khóa tiếng Việt; mức rủi ro
- [ ] Bổ sung danh sách thương hiệu & tên miền chính thức từ email thật; thêm email mẫu

**Giai đoạn 2 — mô hình hiểu tiếng Việt và email**
- [ ] Làm sạch giữ chữ có dấu, TF-IDF theo cụm ký tự (character n-gram)
- [ ] Dữ liệu: email spam/phishing tiếng Anh công khai + email tiếng Việt thu thập dần (qua 👍/👎); đánh giá trên email thật
- [ ] Thử mô hình ngôn ngữ đa ngữ / PhoBERT khi đủ dữ liệu

**Giai đoạn 3 — chạy thật**
- [x] GitHub Actions (chạy pytest)
- [ ] REST API với FastAPI (`POST /analyze` nhận `.eml`), Dockerfile
- [ ] Mặc định không lưu nội dung email; đăng nhập cho trang phản hồi; duyệt feedback trước khi đưa vào train
- [ ] Lưu log/feedback vào SQL database, theo dõi tỉ lệ báo nhầm sau triển khai
