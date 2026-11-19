"""Test đọc email (.eml / nguồn email) và dấu hiệu rủi ro theo cấu trúc email. Chạy: pytest"""
import sys
from email import policy
from email.message import EmailMessage
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from email_parser import Link, parse_eml, looks_like_raw_email  # noqa: E402
from email_risk import (assess_risk, detect_email_signals, is_official, link_issues,  # noqa: E402
                        lookalike_brand, merge_signals, registrable_domain)
from preprocess import strip_accents  # noqa: E402
from signals import detect_signals  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def make_email(from_="Nguyen Van A <a@congty.example>", subject="Chao ban", text="Noi dung", html=None,
               attachments=(), **headers) -> bytes:
    m = EmailMessage()
    for name, value in headers.items():
        m[name.replace("_", "-")] = value
    m["From"], m["To"], m["Subject"] = from_, "ban@example.com", subject
    m.set_content(text)
    if html:
        m.add_alternative(html, subtype="html")
    for filename in attachments:
        m.add_attachment(b"x", maintype="application", subtype="octet-stream", filename=filename)
    return m.as_bytes(policy=policy.SMTP)


def ids(signals):
    return {s["id"] for s in signals}


# ----------------------------------------------------------------- đọc email
def test_parse_basic_headers_and_vietnamese_subject():
    pe = parse_eml(make_email(from_='"Phòng Kế toán" <ketoan@congty.example>', subject="Bảng lương tháng 11",
                              Reply_To="Kế toán <ketoan@congty.example>"))
    assert pe.from_name == "Phòng Kế toán" and pe.from_addr == "ketoan@congty.example"
    assert pe.from_domain == "congty.example" and pe.reply_to_domain == "congty.example"
    assert pe.subject == "Bảng lương tháng 11"
    assert pe.model_text().startswith("Bảng lương tháng 11\nNoi dung")


def test_parse_html_links_keep_visible_text_and_real_target():
    html = ('<html><head><style>.x{}</style><script>var a=1</script></head><body>'
            '<p>Xin chào &amp; cảm ơn</p><a href="https://evil.test/login">https://ngan-hang.example/dang-nhap</a>'
            '<a href="mailto:a@b.c">mail</a><a href="#top">lên đầu</a></body></html>')
    pe = parse_eml(make_email(text="", html=html))
    assert pe.has_html
    assert [(lk.text, lk.host) for lk in pe.links] == [("https://ngan-hang.example/dang-nhap", "evil.test")]
    assert "var a" not in parse_eml(make_email(text="", html=html)).text


def test_html_only_email_uses_html_text_for_model():
    m = EmailMessage()
    m["From"], m["Subject"] = "a@b.example", "Hi"
    m.set_content("<p>Hello <b>there</b></p><p>Second line</p>", subtype="html")
    pe = parse_eml(m.as_bytes())
    assert pe.text == "Hello there\n\nSecond line"


def test_plain_text_email_extracts_links_from_text():
    pe = parse_eml(make_email(text="Xem tai https://bit.ly/abc nhe."))
    assert [lk.host for lk in pe.links] == ["bit.ly"]


def test_parse_attachments_and_auth_uses_topmost_header_only():
    raw = make_email(attachments=["hoa_don.pdf.exe", "anh.jpg"],
                     Authentication_Results="mx.example.net; spf=fail smtp.mailfrom=x; dkim=pass; dmarc=fail")
    # header giả do kẻ gian chèn sẵn nằm BÊN DƯỚI header của máy chủ nhận -> phải bị bỏ qua
    raw = raw.replace(b"From:", b"Authentication-Results: fake; spf=pass; dkim=pass; dmarc=pass\r\nFrom:", 1)
    pe = parse_eml(raw)
    assert pe.auth == {"spf": "fail", "dkim": "pass", "dmarc": "fail"}
    assert [a.filename for a in pe.attachments] == ["hoa_don.pdf.exe", "anh.jpg"]


def test_parse_accepts_pasted_source_text_and_detects_it():
    raw = make_email().decode()
    assert looks_like_raw_email(raw)
    assert parse_eml(raw).from_addr == "a@congty.example"
    assert not looks_like_raw_email("Chào anh,\nFrom: hôm qua em có gửi file")
    assert not looks_like_raw_email("WINNER!! Claim your prize now")


def test_parse_rejects_huge_email():
    with pytest.raises(ValueError):
        parse_eml(b"x" * (16 * 1024 * 1024))


def test_parse_survives_broken_html_and_bad_charset():
    raw = make_email(html="<p>Unclosed <a href='https://ok.example'>link")
    raw = raw.replace(b'charset="utf-8"', b'charset="khong-ton-tai"')
    pe = parse_eml(raw)
    assert pe.from_addr  # không ném lỗi


# ----------------------------------------------------------------- tên miền
@pytest.mark.parametrize("host,expected", [
    ("info.vietcombank.com.vn", "vietcombank.com.vn"), ("a.b.example.com", "example.com"),
    ("Mail.Google.COM.", "google.com"), ("192.168.1.1", "192.168.1.1"), ("sgu.edu.vn", "sgu.edu.vn"),
])
def test_registrable_domain(host, expected):
    assert registrable_domain(host) == expected


@pytest.mark.parametrize("host", [
    "vietcombamk.test", "secure-vietcombank.xyz", "vietcombank.com.vn.dang-nhap.test", "paypai.com",
    "paypa1-security.test", "rnicrosoft.com", "faceb00k.com", "agrlbank.com",
    "việtcombank".encode("idna").decode() + ".com", "pаypal".encode("idna").decode() + ".com",  # "а" Cyrillic
])
def test_lookalike_domains_are_flagged_high(host):
    look = lookalike_brand(host)
    assert look is not None and look[2] == "high", (host, look)


@pytest.mark.parametrize("host", [
    # ngân hàng thật có tên rất gần nhau -> không được báo nhầm là giả mạo
    "abbank.vn", "lpbank.com.vn", "hdbank.com.vn", "seabank.com.vn", "shb.com.vn",
    # tên miền chính thức / tên miền con / tên miền bình thường
    "info.vietcombank.com.vn", "login.microsoftonline.com", "googleapis.com", "office.truong.edu.vn",
    "pineapple.vn", "sgu.edu.vn", "example.com",
])
def test_legit_domains_are_not_flagged(host):
    assert lookalike_brand(host) is None, (host, lookalike_brand(host))


def test_same_name_other_tld_is_only_medium():
    assert lookalike_brand("vietcombank.com")[2] == "medium"
    assert is_official("ebank.vietcombank.com.vn", "Vietcombank")


# ----------------------------------------------------------------- dấu hiệu
def test_display_name_spoof_from_free_mail():
    pe = parse_eml(make_email(from_='"Vietcombank CSKH" <vcb.hotro@gmail.com>'))
    sig = {s["id"]: s for s in detect_email_signals(pe)}
    assert sig["display_spoof"]["severity"] == "high" and "hộp thư miễn phí" in sig["display_spoof"]["detail"]


def test_display_name_brand_from_official_domain_is_fine():
    pe = parse_eml(make_email(from_='"Vietcombank" <noreply@info.vietcombank.com.vn>',
                              Authentication_Results="mx; spf=pass; dkim=pass; dmarc=pass"))
    assert detect_email_signals(pe) == []


def test_display_name_brand_matches_without_accents():
    pe = parse_eml(make_email(from_='"TỔNG CỤC THUẾ" <thongbao@thue-online.test>'))
    assert "display_spoof" in ids(detect_email_signals(pe))


def test_reply_to_mismatch_and_auth_failures():
    pe = parse_eml(make_email(Reply_To="other@gmail.com",
                              Authentication_Results="mx; spf=fail; dkim=fail; dmarc=fail"))
    sig = {s["id"]: s for s in detect_email_signals(pe)}
    assert sig["dmarc_fail"]["severity"] == "high"
    assert sig["auth_fail"]["severity"] == "medium" and "SPF=fail" in sig["auth_fail"]["detail"]
    assert sig["reply_to_mismatch"]["severity"] == "medium"


def test_spf_softfail_alone_is_low():
    pe = parse_eml(make_email(Authentication_Results="mx; spf=softfail; dkim=pass; dmarc=pass"))
    assert [(s["id"], s["severity"]) for s in detect_email_signals(pe)] == [("auth_fail", "low")]


def test_link_issues():
    assert ids_of(link_issues(Link("https://vietcombank.com.vn", "https://x.test/a", "x.test"))) == {"link_mismatch"}
    assert ids_of(link_issues(Link("Bấm vào đây", "http://10.0.0.5/login", "10.0.0.5"))) == {"link_ip"}
    assert ids_of(link_issues(Link("Xem", "https://bit.ly/x", "bit.ly"))) == {"short_link"}
    # chữ hiển thị là URL cùng tên miền -> bình thường
    assert link_issues(Link("www.example.com/a", "https://example.com/a?utm=1", "example.com")) == []
    # chữ hiển thị không phải URL -> không so khớp
    assert link_issues(Link("Đăng nhập", "https://example.com/login", "example.com")) == []


def ids_of(issues):
    return {i[0] for i in issues}


@pytest.mark.parametrize("filename,sid", [
    ("hoa_don.pdf.exe", "attach_double_ext"), ("setup.exe", "attach_exec"), ("bien_ban.html", "attach_html"),
    ("bao_cao.docm", "attach_macro"), ("tai_lieu.zip", "attach_archive"),
])
def test_attachment_signals(filename, sid):
    assert sid in ids(detect_email_signals(parse_eml(make_email(attachments=[filename]))))


def test_safe_attachments_are_ignored():
    assert detect_email_signals(parse_eml(make_email(attachments=["lich_thi.pdf", "anh.jpg", "bao_cao.docx"]))) == []


# ----------------------------------------------------------------- tiếng Việt trong signals.py
def test_vietnamese_keywords_with_and_without_accents():
    with_accents = detect_signals("Tài khoản sẽ bị KHÓA trong 24h. Vui lòng XÁC MINH tại https://x.test ngay lập tức")
    assert {"phishing", "urgency"} <= ids(with_accents)
    phishing = next(s for s in with_accents if s["id"] == "phishing")
    assert "xác minh" in phishing["detail"]  # hiển thị đúng chữ có dấu trong email
    assert {"phishing", "money"} <= ids(detect_signals("Ban da trung thuong, xac minh tai https://x.test"))


def test_vietnamese_keywords_avoid_false_matches():
    assert detect_signals("Hẹn gặp anh lúc 3 giờ chiều mai nhé, pin điện thoại em sắp hết") == []


def test_weak_account_words_with_link_are_not_phishing():
    notice = "Ngân hàng thông báo: số dư tài khoản +1.000.000₫. Chi tiết: https://vietcombank.com.vn"
    sig = detect_signals(notice)
    assert "phishing" not in ids(sig)
    assert assess_risk(sig)["level"] == "low"


def test_strip_accents_keeps_length():
    text = "Xác Minh Đăng nhập TÀI KHOẢN"
    assert strip_accents(text) == "xac minh dang nhap tai khoan" and len(strip_accents(text)) == len(text)


# ----------------------------------------------------------------- tổng hợp
def test_merge_signals_keeps_most_severe_per_id():
    merged = merge_signals([{"id": "a", "severity": "low", "title": "", "detail": ""}],
                           [{"id": "a", "severity": "high", "title": "", "detail": ""},
                            {"id": "b", "severity": "medium", "title": "", "detail": ""}])
    assert [(s["id"], s["severity"]) for s in merged] == [("a", "high"), ("b", "medium")]


def test_assess_risk_levels():
    def sig(sev):
        return {"id": sev, "severity": sev, "title": "", "detail": ""}
    assert assess_risk([])["level"] == "low"
    assert assess_risk([sig("low"), sig("low")])["level"] == "low"
    assert assess_risk([sig("medium")])["level"] == "medium"
    assert assess_risk([sig("high")])["level"] == "high"


@pytest.mark.parametrize("name,level", [
    ("lua_dao_ngan_hang_vi.eml", "high"), ("phishing_paypal_en.eml", "high"), ("hop_le_lich_thi_vi.eml", "low"),
])
def test_sample_emails_end_to_end(name, level):
    pe = parse_eml((SAMPLES / name).read_bytes())
    signals = merge_signals(detect_signals(pe.model_text()), detect_email_signals(pe))
    assert assess_risk(signals)["level"] == level


# ----------------------------------------------------------------- giao diện (HTML thuần, không cần streamlit)
def test_email_ui_components_are_single_line_and_escaped():
    import ui_components as ui
    from email_risk import attachment_issue

    pe = parse_eml(make_email(from_='"<script>alert(1)</script>" <x@evil.test>', subject="<b>Tiêu đề</b>",
                              html='<a href="https://bit.ly/x"><img alt="<i>nút</i>"></a>',
                              attachments=["a.pdf.exe"], Authentication_Results="mx; spf=pass; dmarc=fail"))
    signals = merge_signals(detect_signals(pe.model_text()), detect_email_signals(pe))
    parts = [ui.risk_card(assess_risk(signals)), ui.email_header_card(pe), ui.links_html(pe.links, link_issues),
             ui.attachments_html(pe.attachments, attachment_issue), ui.signals_html(signals)]
    for part in parts:
        assert "\n" not in part and "<script>" not in part and "<b>Tiêu" not in part and "<i>nút" not in part
    assert "DMARC: fail" in parts[1] and "bit.ly" in parts[2] and "đuôi kép" in parts[3]
    assert ui.links_html([], link_issues) == "" and ui.attachments_html([], attachment_issue) == ""
    assert "Không có thông tin" in ui.email_header_card(parse_eml(make_email()))


def test_analyze_and_cli(capsys, monkeypatch):
    from email_risk import analyze, main

    result = analyze((SAMPLES / "phishing_paypal_en.eml").read_bytes())
    assert result["risk"]["level"] == "high" and result["email"].from_domain == "paypa1-security.test"
    monkeypatch.setattr(sys, "argv", ["email_risk.py", str(SAMPLES / "hop_le_lich_thi_vi.eml")])
    main()
    assert "RỦI RO THẤP" in capsys.readouterr().out
