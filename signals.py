"""Phát hiện các dấu hiệu đáng ngờ bằng luật (rule-based), cho cả tiếng Anh và tiếng Việt.

Đây là lớp giải thích BỔ SUNG, độc lập với mô hình ML: giúp người dùng thấy cụ thể email có
link rút gọn, từ khẩn cấp, nhắc đến tiền thưởng, yêu cầu thông tin tài khoản... Kết quả chỉ mang tính
tham khảo, không dùng để quyết định nhãn Spam/Ham.
"""
import re
import unicodedata
from urllib.parse import urlsplit

from preprocess import URL_RE, strip_accents

SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy")

URGENCY_TERMS = ["urgent", "immediately", "act now", "limited time", "final notice", "expires", "expire",
                 "asap", "last chance", "hurry", "today only", "right away"]
MONEY_TERMS = ["free", "win", "winner", "won", "prize", "cash", "bonus", "reward", "gift card", "lottery",
               "congratulations", "guaranteed", "earn", "discount", "offer"]
MONEY_SYMBOLS = "$£€₫"
# Từ khóa "mạnh": đòi thông tin đăng nhập / xác minh -> kèm link thì nghi lừa đảo.
CREDENTIAL_TERMS = ["password", "verify", "verification", "suspended", "locked", "login", "log in", "sign in",
                    "security alert", "unusual activity", "otp"]
# Từ khóa "yếu": rất hay gặp cả trong email hợp lệ (thông báo ngân hàng, tài khoản...) -> chỉ ở mức nhẹ.
WEAK_CREDENTIAL_TERMS = ["account", "bank", "confirm"]

# Tiếng Việt: viết KHÔNG DẤU, so khớp trên văn bản đã bỏ dấu (bắt được cả khi kẻ gian cố tình viết không dấu).
# Tránh từ dễ trùng nghĩa sau khi bỏ dấu (vd "gấp" -> "gap" trùng "gặp").
URGENCY_TERMS_VI = ["khan cap", "ngay lap tuc", "het han", "sap het han", "trong vong 24", "trong 24h",
                    "trong 24 gio", "han chot", "thong bao cuoi", "lan cuoi", "xu ly ngay", "bi dinh chi",
                    "vo hieu hoa"]
MONEY_TERMS_VI = ["trung thuong", "giai thuong", "qua tang", "mien phi", "hoan tien", "nhan thuong",
                  "tien thuong", "khuyen mai", "uu dai", "nop phat", "phi phat", "thanh toan ngay", "giai ngan",
                  "vay tien", "lai suat thap", "chuyen khoan ngay"]
CREDENTIAL_TERMS_VI = ["mat khau", "ma otp", "otp", "xac minh", "xac thuc", "dang nhap", "cap nhat thong tin",
                       "ma pin", "so the", "cvv", "bi khoa", "tam khoa", "hoat dong bat thuong",
                       "canh bao bao mat"]
WEAK_CREDENTIAL_TERMS_VI = ["tai khoan", "ngan hang", "so tai khoan", "cccd", "can cuoc"]

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _compile(terms: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


URGENCY_RE = _compile(URGENCY_TERMS)
MONEY_RE = _compile(MONEY_TERMS)
CREDENTIAL_RE = _compile(CREDENTIAL_TERMS)
WEAK_CREDENTIAL_RE = _compile(WEAK_CREDENTIAL_TERMS)
URGENCY_VI_RE = _compile(URGENCY_TERMS_VI)
MONEY_VI_RE = _compile(MONEY_TERMS_VI)
CREDENTIAL_VI_RE = _compile(CREDENTIAL_TERMS_VI)
WEAK_CREDENTIAL_VI_RE = _compile(WEAK_CREDENTIAL_TERMS_VI)


def _find(text: str, en_re: re.Pattern, vi_re: re.Pattern) -> list[str]:
    """Từ khóa tìm thấy (tiếng Anh trên văn bản gốc, tiếng Việt trên văn bản bỏ dấu), trả về đúng chữ trong email."""
    found = en_re.findall(text)
    nfc = unicodedata.normalize("NFC", text)
    plain = strip_accents(nfc)  # cùng độ dài với nfc -> cắt lại được chữ gốc có dấu
    found += [nfc[m.start():m.end()] for m in vi_re.finditer(plain)]
    return found


def _unique(matches: list[str], limit: int = 4) -> list[str]:
    seen, out = set(), []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:limit]


def url_hostname(url: str) -> str:
    """Hostname (chữ thường, bỏ 'www.') của một URL tìm thấy trong văn bản; '' nếu không phân tích được."""
    url = url.strip().rstrip(".,;:!?)]}>\"'")
    if not re.match(r"[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "http://" + url  # dạng www.example.com
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_shortener(url: str) -> bool:
    """So khớp theo hostname: 'bit.ly' hoặc tên miền con của nó, không so khớp chuỗi con trong cả URL
    (trước đây 'microsoft.com' bị nhận nhầm vì chứa 't.co', '.../bit.ly-guide' cũng bị nhận nhầm)."""
    host = url_hostname(url)
    return any(host == s or host.endswith("." + s) for s in SHORTENERS)


def detect_signals(text: str) -> list[dict]:
    """Trả về danh sách {'id', 'title', 'detail', 'severity': high|medium|low}, nghiêm trọng nhất xếp trước."""
    if not isinstance(text, str) or not text.strip():
        return []
    signals: list[dict] = []

    urls = URL_RE.findall(text)
    short = [u for u in urls if is_shortener(u)]
    if short:
        signals.append({"id": "short_link", "severity": "high", "title": "Link rút gọn",
                        "detail": "Che giấu địa chỉ thật: " + ", ".join(short[:2])})
    plain = len(urls) - len(short)
    if plain > 0:
        # mức nhẹ: email hợp lệ nào cũng có link; link đáng ngờ được đánh giá riêng (rút gọn, giả mạo, IP...)
        signals.append({"id": "links", "severity": "low", "title": f"Chứa {plain} đường link",
                        "detail": "Spam thường kèm link để dẫn người dùng đi nơi khác"})

    cred = _unique(_find(text, CREDENTIAL_RE, CREDENTIAL_VI_RE))
    weak = _unique(_find(text, WEAK_CREDENTIAL_RE, WEAK_CREDENTIAL_VI_RE))
    if cred and urls:
        signals.append({"id": "phishing", "severity": "high", "title": "Nghi ngờ lừa đảo tài khoản",
                        "detail": "Yêu cầu thông tin đăng nhập / xác minh (" + ", ".join(cred) + ") kèm đường link"})
    elif cred or weak:
        signals.append({"id": "credential", "severity": "low", "title": "Nhắc đến tài khoản / mật khẩu",
                        "detail": ", ".join(_unique(cred + weak))})

    urgency = _unique(_find(text, URGENCY_RE, URGENCY_VI_RE))
    if urgency:
        signals.append({"id": "urgency", "severity": "medium", "title": "Giọng điệu khẩn cấp",
                        "detail": ", ".join(urgency)})

    money = _unique(_find(text, MONEY_RE, MONEY_VI_RE))
    symbols = sum(text.count(s) for s in MONEY_SYMBOLS)
    if money or symbols:
        detail = ", ".join(money + (["ký hiệu tiền tệ"] if symbols else []))
        # chỉ có ký hiệu tiền tệ (hóa đơn, biến động số dư...) thì nhẹ; có từ khóa thưởng/khuyến mãi thì đáng chú ý
        signals.append({"id": "money", "severity": "medium" if money else "low",
                        "title": "Nhắc đến tiền / giải thưởng", "detail": detail})

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 15 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        signals.append({"id": "caps", "severity": "medium", "title": "Viết HOA quá nhiều",
                        "detail": "Hơn một nửa số chữ cái là chữ in hoa"})

    if text.count("!") >= 3:
        signals.append({"id": "exclaim", "severity": "low", "title": "Nhiều dấu chấm than",
                        "detail": f"{text.count('!')} dấu !"})

    return sorted(signals, key=lambda s: SEVERITY_ORDER[s["severity"]])
