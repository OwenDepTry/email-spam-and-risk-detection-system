"""Tiền xử lý văn bản và đặc trưng bổ sung cho Spam Email Detector.

Các hàm ở đây được viết ở mức module (không dùng lambda) để pipeline sklearn
có thể lưu bằng joblib và nạp lại an toàn.
"""
import html
import re

import numpy as np

# Danh sách stopwords tiếng Anh nhúng sẵn -> không cần tải dữ liệu NLTK, chạy offline được.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

CURRENCY_SYMBOLS = "$£€"

# Tên các đặc trưng bổ sung (theo đúng thứ tự cột trả về của extra_features)
EXTRA_FEATURE_NAMES = [
    "url_count",        # số link
    "exclamation_count",  # số dấu !
    "upper_ratio",      # tỉ lệ chữ IN HOA
    "digit_ratio",      # tỉ lệ chữ số
    "log_length",       # độ dài (log)
    "currency_count",   # số ký hiệu tiền tệ
]


def clean_text(text: str) -> str:
    """Làm sạch text: lowercase, thay URL/email/số bằng token, bỏ ký tự đặc biệt, bỏ stopwords."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()         # "&lt;#&gt;" -> "<#>" (tránh sinh ra từ rác "lt", "gt")
    text = URL_RE.sub(" urltoken ", text)      # giữ lại tín hiệu "có link" - đặc trưng quan trọng của spam
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in MULTI_SPACE_RE.sub(" ", text).split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def clean_series(texts) -> list[str]:
    """Áp dụng clean_text cho cả danh sách/Series (dùng trong FunctionTransformer)."""
    return [clean_text(t) for t in texts]


def extra_features(texts) -> np.ndarray:
    """Đặc trưng thủ công tính trên văn bản GỐC (trước khi làm sạch). Tất cả giá trị >= 0."""
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        letters = [c for c in t if c.isalpha()]
        rows.append([
            len(URL_RE.findall(t.lower())),
            t.count("!"),
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            sum(c.isdigit() for c in t) / max(len(t), 1),
            float(np.log1p(len(t))),
            sum(t.count(s) for s in CURRENCY_SYMBOLS),
        ])
    return np.array(rows, dtype=float).reshape(len(rows), len(EXTRA_FEATURE_NAMES))


if __name__ == "__main__":
    demo = "WINNER!! You have won $1000. Click http://bit.ly/claim-now to get your prize!!!"
    print(clean_text(demo))
    print(dict(zip(EXTRA_FEATURE_NAMES, extra_features([demo])[0].round(3))))
