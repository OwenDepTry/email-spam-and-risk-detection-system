"""Đọc email thật: file .eml hoặc "nguồn email" dán vào (Gmail: ⋮ → Hiển thị bản gốc / Show original).

Chỉ dùng thư viện chuẩn của Python (email, html.parser), không cần cài thêm gì.
Kết quả là một ParsedEmail gồm: người gửi, Reply-To, tiêu đề, kết quả SPF/DKIM/DMARC, nội dung dạng văn bản,
danh sách link (chữ hiển thị + địa chỉ thật) và danh sách file đính kèm. Module này KHÔNG đánh giá rủi ro,
việc đó nằm ở email_risk.py.
"""
import re
from dataclasses import dataclass, field
from email import message_from_bytes, message_from_string, policy
from email.message import EmailMessage
from email.utils import getaddresses
from html.parser import HTMLParser

from preprocess import URL_RE
from signals import url_hostname

MAX_EMAIL_BYTES = 15 * 1024 * 1024  # file .eml lớn hơn mức này bị từ chối (chủ yếu do file đính kèm)
MAX_LINKS = 200

# Các header thường gặp ở đầu một email thô; dùng để nhận ra khi người dùng dán cả nguồn email.
_RAW_HEADER_RE = re.compile(
    r"^(received|return-path|delivered-to|from|to|subject|date|message-id|mime-version|"
    r"content-type|authentication-results|reply-to|dkim-signature|arc-[a-z-]+|x-[a-z0-9-]+):",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(r"\b(spf|dkim|dmarc)\s*=\s*([a-z]+)", re.IGNORECASE)
_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}


@dataclass
class Link:
    text: str   # chữ người dùng nhìn thấy
    href: str   # địa chỉ thật khi bấm vào
    host: str   # hostname của href ('' nếu không phân tích được)


@dataclass
class Attachment:
    filename: str
    content_type: str
    size: int


@dataclass
class ParsedEmail:
    subject: str = ""
    from_name: str = ""
    from_addr: str = ""
    reply_to: str = ""
    return_path: str = ""
    to: str = ""
    date: str = ""
    auth: dict = field(default_factory=dict)  # {'spf': 'pass'|'fail'|..., 'dkim': ..., 'dmarc': ...}
    text: str = ""                            # nội dung dạng văn bản (từ text/plain, hoặc chuyển từ HTML)
    has_html: bool = False
    links: list[Link] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)

    @property
    def from_domain(self) -> str:
        return domain_of(self.from_addr)

    @property
    def reply_to_domain(self) -> str:
        return domain_of(self.reply_to)

    def model_text(self) -> str:
        """Văn bản đưa vào mô hình ML: tiêu đề + nội dung (giống cách nhập tay trên giao diện)."""
        return (self.subject.strip() + "\n" + self.text.strip()).strip()


def domain_of(addr: str) -> str:
    addr = (addr or "").strip().strip("<>").lower()
    return addr.rsplit("@", 1)[1] if "@" in addr else ""


# ----------------------------------------------------------------------------- HTML -> văn bản + link
class _HTMLExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip = 0           # đang ở trong <script>/<style>/<head>
        self._href: str | None = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "title"):
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._anchor = []
        elif tag == "img" and self._href is not None:
            alt = dict(attrs).get("alt")
            if alt:
                self._anchor.append(alt)
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "title"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._href is not None:
            self.links.append((" ".join("".join(self._anchor).split()), self._href.strip()))
            self._href = None
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._href is not None:
            self._anchor.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text_and_links(html: str) -> tuple[str, list[tuple[str, str]]]:
    parser = _HTMLExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - HTML hỏng: lấy được phần nào hay phần đó
        pass
    return parser.text(), parser.links


def _make_link(text: str, href: str) -> Link | None:
    href = href.strip()
    if not href or href.lower().startswith(("mailto:", "tel:", "#", "cid:", "javascript:")):
        return None
    return Link(text=text.strip(), href=href, host=url_hostname(href))


def links_from_text(text: str) -> list[Link]:
    out = []
    for url in URL_RE.findall(text or ""):
        link = _make_link(url, url)
        if link:
            out.append(link)
    return out


# ----------------------------------------------------------------------------- đọc email
def looks_like_raw_email(text: str) -> bool:
    """True nếu văn bản dán vào giống nguồn email (bắt đầu bằng các dòng header như From:, Received:...)."""
    if not isinstance(text, str):
        return False
    head = text.lstrip()[:5000].split("\n\n", 1)[0].splitlines()[:40]
    hits = {m.group(1).lower() for line in head if (m := _RAW_HEADER_RE.match(line))}
    return len(hits) >= 2 and ("from" in hits or "received" in hits)


def _header(msg: EmailMessage, name: str) -> str:
    try:
        value = msg.get(name)
    except Exception:  # noqa: BLE001 - header sai định dạng: bỏ qua thay vì làm hỏng cả email
        value = ""
    return " ".join(str(value or "").split())


def _parse_auth(msg: EmailMessage) -> dict:
    """SPF/DKIM/DMARC do MÁY CHỦ NHẬN ghi vào header Authentication-Results.

    Chỉ đọc header Authentication-Results ĐẦU TIÊN (trên cùng): đó là header do máy chủ nhận cuối cùng
    (ví dụ Gmail) thêm vào. Kẻ gian có thể tự chèn header giả ở bên dưới, nên không tin các header sau.
    """
    auth = {}
    headers = msg.get_all("Authentication-Results") or []
    if headers:
        for key, value in _AUTH_RE.findall(str(headers[0])):
            key, value = key.lower(), value.lower()
            # có nhiều chữ ký DKIM: chỉ cần một chữ ký hợp lệ là đạt
            if key == "dkim" and auth.get("dkim") == "pass":
                continue
            auth[key] = value
    if "spf" not in auth:
        received_spf = msg.get_all("Received-SPF") or []
        if received_spf:
            first = str(received_spf[0]).strip().split()
            if first:
                auth["spf"] = first[0].lower()
    return auth


def _part_text(part) -> str:
    try:
        return part.get_content()
    except Exception:  # noqa: BLE001 - bảng mã lạ / nội dung hỏng
        payload = part.get_payload(decode=True) or b""
        if isinstance(payload, str):
            return payload
        try:
            return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        except LookupError:  # tên bảng mã không tồn tại -> thử UTF-8
            return payload.decode("utf-8", errors="replace")


def parse_message(msg: EmailMessage) -> ParsedEmail:
    pe = ParsedEmail(
        subject=_header(msg, "Subject"),
        reply_to=", ".join(a for _, a in getaddresses([_header(msg, "Reply-To")]) if a),
        return_path=_header(msg, "Return-Path").strip("<>"),
        to=_header(msg, "To"),
        date=_header(msg, "Date"),
        auth=_parse_auth(msg),
    )
    from_pairs = getaddresses([_header(msg, "From")])
    if from_pairs:
        pe.from_name, pe.from_addr = from_pairs[0][0].strip(), from_pairs[0][1].strip()

    plain_parts, html_parts = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        disposition = (part.get_content_disposition() or "").lower()
        ctype = part.get_content_type()
        if filename or disposition == "attachment":
            payload = part.get_payload(decode=True) or b""
            pe.attachments.append(Attachment(filename=str(filename or "(không tên)"), content_type=ctype,
                                             size=len(payload)))
        elif ctype == "text/plain":
            plain_parts.append(_part_text(part))
        elif ctype == "text/html":
            html_parts.append(_part_text(part))

    raw_links: list[tuple[str, str]] = []
    if html_parts:
        pe.has_html = True
        html_text, raw_links = html_to_text_and_links("\n".join(html_parts))
    else:
        html_text = ""
    pe.text = "\n\n".join(p.strip() for p in plain_parts if p.strip()) or html_text

    if pe.has_html:
        # Có bản HTML thì đó là thứ người dùng thấy và bấm vào (Gmail, Outlook...): chỉ lấy link từ <a href>.
        # Không lấy URL dạng chữ (kể cả trong bản text/plain): chữ hiển thị "https://ngan-hang-that" không phải
        # nơi link thật sự mở ra.
        links = [lk for t, h in raw_links if (lk := _make_link(t, h))]
    else:
        links = links_from_text(pe.text)
    pe.links = links[:MAX_LINKS]
    return pe


def parse_eml(data: bytes | str) -> ParsedEmail:
    """Đọc nội dung file .eml (bytes) hoặc nguồn email dán vào (str)."""
    size = len(data.encode("utf-8", errors="ignore")) if isinstance(data, str) else len(data)
    if size > MAX_EMAIL_BYTES:
        raise ValueError(f"Email quá lớn ({size / 1024 / 1024:.1f} MB, tối đa {MAX_EMAIL_BYTES // 1024 // 1024} MB).")
    if isinstance(data, str):
        msg = message_from_string(data.lstrip(), policy=policy.default)
    else:
        msg = message_from_bytes(data.lstrip(), policy=policy.default)
    return parse_message(msg)
