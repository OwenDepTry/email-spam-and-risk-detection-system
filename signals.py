"""Phát hiện các dấu hiệu đáng ngờ bằng luật (rule-based).

Đây là lớp giải thích BỔ SUNG, độc lập với mô hình ML: giúp người dùng thấy cụ thể email có
link rút gọn, từ khẩn cấp, nhắc đến tiền thưởng, yêu cầu thông tin tài khoản... Kết quả chỉ mang tính
tham khảo, không dùng để quyết định nhãn Spam/Ham.
"""
import re
from urllib.parse import urlsplit

from preprocess import URL_RE

SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy")

URGENCY_TERMS = ["urgent", "immediately", "act now", "limited time", "final notice", "expires", "expire",
                 "asap", "last chance", "hurry", "today only", "right away"]
MONEY_TERMS = ["free", "win", "winner", "won", "prize", "cash", "bonus", "reward", "gift card", "lottery",
               "congratulations", "guaranteed", "earn", "discount", "offer"]
MONEY_SYMBOLS = "$£€"
CREDENTIAL_TERMS = ["password", "verify", "verification", "account", "suspended", "locked", "login",
                    "log in", "bank", "confirm", "security alert", "unusual activity"]

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _compile(terms: list[str]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


URGENCY_RE = _compile(URGENCY_TERMS)
MONEY_RE = _compile(MONEY_TERMS)
CREDENTIAL_RE = _compile(CREDENTIAL_TERMS)


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
        signals.append({"id": "links", "severity": "medium", "title": f"Chứa {plain} đường link",
                        "detail": "Spam thường kèm link để dẫn người dùng đi nơi khác"})

    cred = _unique(CREDENTIAL_RE.findall(text))
    if cred and urls:
        signals.append({"id": "phishing", "severity": "high", "title": "Nghi ngờ lừa đảo tài khoản",
                        "detail": "Yêu cầu thông tin tài khoản (" + ", ".join(cred) + ") kèm đường link"})
    elif cred:
        signals.append({"id": "credential", "severity": "low", "title": "Nhắc đến tài khoản / mật khẩu",
                        "detail": ", ".join(cred)})

    urgency = _unique(URGENCY_RE.findall(text))
    if urgency:
        signals.append({"id": "urgency", "severity": "medium", "title": "Giọng điệu khẩn cấp",
                        "detail": ", ".join(urgency)})

    money = _unique(MONEY_RE.findall(text))
    symbols = sum(text.count(s) for s in MONEY_SYMBOLS)
    if money or symbols:
        detail = ", ".join(money + (["ký hiệu tiền tệ"] if symbols else []))
        signals.append({"id": "money", "severity": "medium", "title": "Nhắc đến tiền / giải thưởng",
                        "detail": detail})

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 15 and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        signals.append({"id": "caps", "severity": "medium", "title": "Viết HOA quá nhiều",
                        "detail": "Hơn một nửa số chữ cái là chữ in hoa"})

    if text.count("!") >= 3:
        signals.append({"id": "exclaim", "severity": "low", "title": "Nhiều dấu chấm than",
                        "detail": f"{text.count('!')} dấu !"})

    return sorted(signals, key=lambda s: SEVERITY_ORDER[s["severity"]])
