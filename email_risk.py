"""Dấu hiệu rủi ro từ CẤU TRÚC email (người gửi, xác thực, link, file đính kèm) và mức rủi ro tổng hợp.

Bổ sung cho signals.py (vốn chỉ nhìn nội dung chữ). Các dấu hiệu ở đây khó che giấu hơn nhiều với email lừa đảo:
kẻ gian có thể viết nội dung rất "sạch", nhưng khó gửi từ đúng tên miền của ngân hàng hay qua được DMARC.

Mọi kết quả là luật (rule-based), có giải thích, độc lập với mô hình ML. Định dạng mỗi dấu hiệu giống signals.py:
{'id', 'severity': high|medium|low, 'title', 'detail'}.
"""
import ipaddress
import re

from email_parser import Attachment, Link, ParsedEmail
from preprocess import strip_accents
from signals import SEVERITY_ORDER, SHORTENERS, url_hostname

# Thương hiệu hay bị giả danh -> (từ khóa nhận diện trong tên hiển thị, tên miền chính thức).
# Tên miền con của tên miền chính thức đều được chấp nhận (vd: info.vietcombank.com.vn).
# LƯU Ý: thương hiệu thật có thể gửi qua tên miền khác (dịch vụ gửi mail, tên miền phụ). Nếu gặp báo nhầm,
# bổ sung tên miền vào danh sách thay vì bỏ luật.
BRANDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Vietcombank": (("vietcombank",), ("vietcombank.com.vn",)),
    "Techcombank": (("techcombank",), ("techcombank.com.vn", "techcombank.com")),
    "BIDV": (("bidv",), ("bidv.com.vn",)),
    "VietinBank": (("vietinbank",), ("vietinbank.vn",)),
    "Agribank": (("agribank",), ("agribank.com.vn",)),
    "MB Bank": (("mbbank", "mb bank"), ("mbbank.com.vn",)),
    "VPBank": (("vpbank",), ("vpbank.com.vn",)),
    "ACB": (("acb", "ngan hang a chau"), ("acb.com.vn",)),
    "Sacombank": (("sacombank",), ("sacombank.com.vn",)),
    "TPBank": (("tpbank",), ("tpb.vn", "tpbank.vn")),
    "MoMo": (("momo",), ("momo.vn",)),
    "ZaloPay": (("zalopay",), ("zalopay.vn",)),
    "Shopee": (("shopee",), ("shopee.vn", "shopee.com")),
    "Lazada": (("lazada",), ("lazada.vn", "lazada.com")),
    "Tiki": (("tiki",), ("tiki.vn",)),
    "Viettel Post": (("viettel post", "viettelpost"), ("viettelpost.com.vn", "viettelpost.vn")),
    "Giao Hàng Nhanh": (("giao hang nhanh",), ("ghn.vn",)),
    "Tổng cục Thuế": (("tong cuc thue", "cuc thue"), ("gdt.gov.vn",)),
    "PayPal": (("paypal",), ("paypal.com",)),
    "Apple": (("apple", "icloud"), ("apple.com", "icloud.com")),
    "Microsoft": (("microsoft", "office 365", "microsoft 365"), ("microsoft.com", "outlook.com", "office.com",
                                                                 "microsoftonline.com")),
    "Google": (("google", "gmail"), ("google.com", "gmail.com")),
    "Facebook": (("facebook", "meta"), ("facebook.com", "facebookmail.com", "meta.com")),
    "Amazon": (("amazon",), ("amazon.com",)),
    "Netflix": (("netflix",), ("netflix.com",)),
    "DHL": (("dhl",), ("dhl.com",)),
}

FREE_MAIL = {"gmail.com", "googlemail.com", "yahoo.com", "yahoo.com.vn", "outlook.com", "hotmail.com", "live.com",
             "icloud.com", "aol.com", "proton.me", "protonmail.com", "gmx.com", "mail.com", "zoho.com"}

# Đuôi tên miền nhiều phần (để lấy đúng "tên miền đăng ký": vietcombank.com.vn chứ không phải com.vn)
MULTI_PART_SUFFIXES = {
    "com.vn", "net.vn", "org.vn", "edu.vn", "gov.vn", "ac.vn", "info.vn", "biz.vn", "name.vn", "pro.vn",
    "health.vn", "int.vn", "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp",
    "com.sg", "com.cn", "co.kr", "com.br", "co.in", "com.my", "co.th", "co.id",
}

DANGEROUS_EXT = {"exe", "scr", "com", "pif", "bat", "cmd", "js", "jse", "vbs", "vbe", "wsf", "ps1", "jar", "msi",
                 "hta", "lnk", "iso", "img", "vhd", "apk", "reg", "cpl"}
HTML_EXT = {"html", "htm", "shtml", "svg", "xhtml"}           # trang đăng nhập giả mở ngay trên máy
MACRO_EXT = {"docm", "xlsm", "pptm", "dotm", "xltm", "xlam", "xls", "doc"}  # có thể chứa macro
ARCHIVE_EXT = {"zip", "rar", "7z", "gz", "tar", "cab", "ace"}  # hay dùng để giấu file độc

_CONFUSABLES = [("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("i", "l"), ("3", "e"), ("5", "s"), ("@", "a")]
# Tên miền là từ thông dụng: không dùng để so khớp "chứa tên thương hiệu" (office.truong.edu.vn không phải Microsoft)
_GENERIC_LABELS = {"office", "outlook", "meta", "gmail"}
# Chỉ so độ giống (khác 1-2 ký tự) với tên đủ dài: tên ngắn của các ngân hàng thật rất gần nhau
# (abbank/mbbank, lpbank/tpbank, hdbank/mbbank...) nên so độ giống sẽ báo nhầm.
_FUZZY_MIN_LEN = 7
_URLISH_RE = re.compile(r"^(https?://)?(www\.)?[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", re.IGNORECASE)

SEVERITY_POINTS = {"high": 3.0, "medium": 1.0, "low": 0.25}


# ----------------------------------------------------------------------------- tiện ích tên miền
def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def registrable_domain(host: str) -> str:
    """'info.vietcombank.com.vn' -> 'vietcombank.com.vn'; 'a.b.example.com' -> 'example.com'."""
    host = (host or "").strip().strip(".").lower()
    if not host or is_ip(host):
        return host
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _label(domain: str) -> str:
    """Phần tên của tên miền đăng ký: 'vietcombank.com.vn' -> 'vietcombank'."""
    return registrable_domain(domain).split(".")[0]


def _levenshtein(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _official_domains() -> dict[str, str]:
    return {d: brand for brand, (_, domains) in BRANDS.items() for d in domains}


def is_official(host: str, brand: str | None = None) -> bool:
    """host thuộc (tên miền con của) tên miền chính thức của brand (hoặc của bất kỳ thương hiệu nào)."""
    reg = registrable_domain(host)
    domains = BRANDS[brand][1] if brand else _official_domains()
    return any(reg == registrable_domain(d) for d in domains)


def _normalize_confusables(label: str) -> str:
    for src, dst in _CONFUSABLES:
        label = label.replace(src, dst)
    return label


# Ký tự Cyrillic / Hy Lạp nhìn giống hệt chữ Latin (pаypal với "а" Cyrillic)
_HOMOGLYPHS = str.maketrans("асеорхуіјԁɡѕԛաοατνκιρ", "aceopxyijdgsqwoatvkip")


def _decode_punycode(label: str) -> str:
    if not label.startswith("xn--"):
        return ""
    try:
        text = label.encode("ascii").decode("idna").lower().translate(_HOMOGLYPHS)
    except UnicodeError:
        return ""
    return re.sub(r"[^a-z0-9]", "", strip_accents(text))


def lookalike_brand(host: str) -> tuple[str, str, str] | None:
    """Nếu host trông giống tên miền của một thương hiệu nhưng KHÔNG phải tên miền thật
    -> (thương hiệu, lý do, mức độ). Ngược lại trả về None."""
    host = (host or "").lower()
    if not host or is_ip(host) or is_official(host):
        return None
    reg = registrable_domain(host)
    label = reg.split(".")[0]
    tokens = set(re.split(r"[-.]", host))
    decoded = _decode_punycode(label)  # 'xn--vitcombank-d4a' -> 'vietcombank' (bỏ dấu) để so như chữ thường
    for official, brand in _official_domains().items():
        off_label = _label(official)
        if len(off_label) < 4:
            continue
        if label == off_label:
            return (brand, f"cùng tên '{off_label}' nhưng khác đuôi tên miền ({reg} ≠ {official})", "medium")
        if off_label not in _GENERIC_LABELS and (
                off_label in tokens or (len(off_label) >= 8 and off_label in host)):
            return (brand, f"chứa '{off_label}' nhưng tên miền thật là {reg}", "high")
        norm_off = _normalize_confusables(off_label)
        if _normalize_confusables(label) == norm_off:
            return (brand, f"'{label}' dùng ký tự trông giống '{off_label}'", "high")
        tricky = next((t for t in tokens if t != off_label and _normalize_confusables(t) == norm_off), None)
        if tricky:  # paypa1-security.test, secure-g00gle.com
            return (brand, f"'{tricky}' dùng ký tự trông giống '{off_label}' (tên miền thật là {reg})", "high")
        if decoded and (decoded == off_label or (len(off_label) >= _FUZZY_MIN_LEN
                                                 and _levenshtein(decoded, off_label, 2) <= 2)):
            return (brand, f"tên miền quốc tế (punycode) hiển thị gần giống '{off_label}'", "high")
        if len(off_label) >= _FUZZY_MIN_LEN and 0 < _levenshtein(label, off_label, 2) <= 2:
            return (brand, f"'{label}' gần giống '{off_label}' (khác 1-2 ký tự)", "high")
    if host.startswith("xn--") or ".xn--" in host:
        return ("", "tên miền dùng ký tự quốc tế (punycode); tên miền tiếng Việt hợp lệ cũng có dạng này, "
                    "hãy kiểm tra kỹ", "low")
    return None


def _brand_in_display_name(name: str) -> str | None:
    plain = f" {re.sub(r'[^a-z0-9.]+', ' ', strip_accents(name))} "
    for brand, (keywords, _) in BRANDS.items():
        if any(f" {kw} " in plain for kw in keywords):
            return brand
    return None


def _ext(filename: str) -> list[str]:
    return [p.lower() for p in filename.strip().split(".")[1:]]


# ----------------------------------------------------------------------------- dấu hiệu
def _sig(sid: str, severity: str, title: str, detail: str) -> dict:
    return {"id": sid, "severity": severity, "title": title, "detail": detail}


def _sender_signals(pe: ParsedEmail) -> list[dict]:
    out = []
    domain = pe.from_domain
    brand = _brand_in_display_name(pe.from_name)
    if brand and domain and not is_official(domain, brand):
        where = "hộp thư miễn phí" if domain in FREE_MAIL else "tên miền khác"
        out.append(_sig("display_spoof", "high", f"Giả danh {brand}",
                        f"Tên hiển thị “{pe.from_name}” nhưng gửi từ {where}: {pe.from_addr}"))
    if domain:
        look = lookalike_brand(domain)
        if look:
            brand_name, reason, severity = look
            out.append(_sig("sender_lookalike", severity, "Tên miền người gửi đáng ngờ",
                            f"{domain}: {reason}" + (f" (giống {brand_name})" if brand_name else "")))
    rt = pe.reply_to_domain
    if rt and domain and registrable_domain(rt) != registrable_domain(domain):
        sev = "medium" if rt in FREE_MAIL or brand else "low"
        out.append(_sig("reply_to_mismatch", sev, "Trả lời sẽ gửi đến địa chỉ khác",
                        f"Reply-To là {pe.reply_to}, khác tên miền người gửi ({domain})"))
    return out


def _auth_signals(pe: ParsedEmail) -> list[dict]:
    a = pe.auth
    out = []
    if a.get("dmarc") == "fail":
        out.append(_sig("dmarc_fail", "high", "Không qua kiểm tra DMARC",
                        f"Máy chủ nhận xác nhận email không thật sự đến từ {pe.from_domain or 'tên miền người gửi'}"))
    bad = []
    if a.get("spf") in ("fail", "softfail", "permerror"):
        bad.append(f"SPF={a['spf']}")
    if a.get("dkim") in ("fail", "permerror"):
        bad.append(f"DKIM={a['dkim']}")
    if bad:
        # SPF softfail hay gặp cả với email thật bị chuyển tiếp nên chỉ ở mức nhẹ
        sev = "low" if bad == ["SPF=softfail"] else "medium"
        out.append(_sig("auth_fail", sev, "Xác thực người gửi không đạt", ", ".join(bad)))
    return out


def link_issues(lk: Link) -> list[tuple[str, str, str]]:
    """Vấn đề của một link: [(id, mức độ, mô tả ngắn)]. Rỗng nếu không thấy gì bất thường."""
    host = lk.host
    if not host:
        return []
    issues = []
    text = lk.text.strip().lower()
    if _URLISH_RE.match(text) and text != lk.href.lower():
        shown = url_hostname(text)
        if shown and registrable_domain(shown) != registrable_domain(host):
            issues.append(("link_mismatch", "high", f"hiện “{shown}” nhưng mở {host}"))
    if is_ip(host):
        issues.append(("link_ip", "high", f"mở thẳng địa chỉ IP {host}"))
    if any(host == s or host.endswith("." + s) for s in SHORTENERS):
        issues.append(("short_link", "high", f"link rút gọn {host}"))
    look = lookalike_brand(host)
    if look:
        issues.append(("link_lookalike", look[2], f"{host}: {look[1]}"))
    return issues


_LINK_TITLES = {
    "link_mismatch": "Link hiển thị khác địa chỉ thật",
    "link_lookalike": "Link đến tên miền đáng ngờ",
    "link_ip": "Link trỏ thẳng đến địa chỉ IP",
    "short_link": "Link rút gọn",
}


def _link_signals(pe: ParsedEmail) -> list[dict]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for lk in pe.links:
        for sid, severity, desc in link_issues(lk):
            grouped.setdefault(sid, []).append((severity, desc))
    out = []
    for sid, items in grouped.items():
        severity = min((sev for sev, _ in items), key=SEVERITY_ORDER.get)
        descs = list(dict.fromkeys(d for _, d in items))  # bỏ trùng, giữ thứ tự
        detail = "; ".join(descs[:3]) + (f" (+{len(descs) - 3} link khác)" if len(descs) > 3 else "")
        out.append(_sig(sid, severity, _LINK_TITLES[sid], detail))
    return out


def attachment_issue(att: Attachment) -> tuple[str, str, str, str] | None:
    """(id, mức độ, tiêu đề, mô tả) nếu file đính kèm đáng ngờ, ngược lại None."""
    exts = _ext(att.filename)
    last = exts[-1] if exts else ""
    if len(exts) >= 2 and last in DANGEROUS_EXT | HTML_EXT:
        return ("attach_double_ext", "high", "File đính kèm đuôi kép",
                f"{att.filename}: trông như .{exts[-2]} nhưng thật ra là .{last}")
    if last in DANGEROUS_EXT:
        return ("attach_exec", "high", "File đính kèm có thể chạy được", att.filename)
    if last in HTML_EXT:
        return ("attach_html", "high", "File đính kèm là trang web",
                f"{att.filename}: hay dùng để mở trang đăng nhập giả ngay trên máy")
    if last in MACRO_EXT:
        return ("attach_macro", "medium", "File Office có thể chứa macro", att.filename)
    if last in ARCHIVE_EXT:
        return ("attach_archive", "low", "File nén", f"{att.filename}: hãy kiểm tra kỹ bên trong")
    return None


def _attachment_signals(pe: ParsedEmail) -> list[dict]:
    grouped: dict[str, dict] = {}
    for att in pe.attachments:
        issue = attachment_issue(att)
        if issue:
            sid, severity, title, desc = issue
            if sid in grouped:
                grouped[sid]["detail"] += f"; {desc}"
            else:
                grouped[sid] = _sig(sid, severity, title, desc)
    return list(grouped.values())


def detect_email_signals(pe: ParsedEmail) -> list[dict]:
    """Dấu hiệu từ cấu trúc email, nghiêm trọng nhất xếp trước."""
    if pe is None:
        return []
    signals = _sender_signals(pe) + _auth_signals(pe) + _link_signals(pe) + _attachment_signals(pe)
    return sorted(signals, key=lambda s: SEVERITY_ORDER[s["severity"]])


def merge_signals(*groups: list[dict]) -> list[dict]:
    """Gộp nhiều danh sách dấu hiệu, bỏ trùng theo id (giữ bản nghiêm trọng hơn), sắp xếp theo mức độ."""
    best: dict[str, dict] = {}
    for group in groups:
        for s in group or []:
            cur = best.get(s["id"])
            if cur is None or SEVERITY_ORDER[s["severity"]] < SEVERITY_ORDER[cur["severity"]]:
                best[s["id"]] = s
    return sorted(best.values(), key=lambda s: SEVERITY_ORDER[s["severity"]])


def assess_risk(signals: list[dict]) -> dict:
    """Mức rủi ro tổng hợp theo luật: 'high' nếu có dấu hiệu nghiêm trọng hoặc nhiều dấu hiệu đáng chú ý.

    Đây là quy tắc minh bạch, CHƯA được hiệu chỉnh bằng dữ liệu thật; nên đọc kèm danh sách lý do.
    """
    score = sum(SEVERITY_POINTS[s["severity"]] for s in signals)
    n_high = sum(s["severity"] == "high" for s in signals)
    if n_high or score >= 3:
        level = "high"
    elif score >= 1:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "score": round(score, 2), "n_signals": len(signals), "n_high": n_high}


def analyze(data: bytes | str) -> dict:
    """Đọc email (.eml bytes hoặc nguồn email dạng chữ) và trả về {'email', 'signals', 'risk'}."""
    from email_parser import parse_eml  # noqa: PLC0415
    from signals import detect_signals  # noqa: PLC0415

    pe = parse_eml(data)
    signals = merge_signals(detect_signals(pe.model_text()), detect_email_signals(pe))
    return {"email": pe, "signals": signals, "risk": assess_risk(signals)}


def main() -> None:
    import argparse  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Kiểm tra dấu hiệu lừa đảo của một hoặc nhiều file .eml")
    parser.add_argument("files", nargs="+", help="Đường dẫn file .eml")
    args = parser.parse_args()
    labels = {"high": "RỦI RO CAO", "medium": "CẦN THẬN TRỌNG", "low": "RỦI RO THẤP"}
    for path in args.files:
        result = analyze(Path(path).read_bytes())
        pe, risk = result["email"], result["risk"]
        print(f"\n{path}\n  Người gửi: {pe.from_name} <{pe.from_addr}> | Tiêu đề: {pe.subject}")
        print(f"  Xác thực: {pe.auth or 'không có'} | {len(pe.links)} link, {len(pe.attachments)} file đính kèm")
        print(f"  => {labels[risk['level']]} ({risk['n_signals']} dấu hiệu, {risk['n_high']} nghiêm trọng)")
        for s in result["signals"]:
            print(f"     [{s['severity']:<6}] {s['title']}: {s['detail']}")


if __name__ == "__main__":
    main()
