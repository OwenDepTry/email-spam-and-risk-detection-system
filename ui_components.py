"""Thành phần giao diện dạng HTML/CSS cho app.py (thuần Python, không phụ thuộc streamlit nên test được).

Dùng với st.markdown(html, unsafe_allow_html=True). Lưu ý: Markdown coi HTML có dòng trống hoặc
thụt đầu dòng 4 ký tự là code block, nên mọi hàm ở đây trả về chuỗi MỘT DÒNG và tự escape dữ liệu đầu vào.
Màu nền dùng rgba trung tính + màu chữ kế thừa nên hiển thị tốt ở cả chế độ sáng và tối.
"""
import html
import re

SPAM_COLOR = "#ef4444"
HAM_COLOR = "#22c55e"
ACCENT = "#6366f1"
SEVERITY_COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#94a3b8"}
SEVERITY_LABELS = {"high": "Nghiêm trọng", "medium": "Đáng chú ý", "low": "Nhẹ"}

_CSS = """
.sd-hero{background:linear-gradient(135deg,#4f46e5,#7c3aed 55%,#db2777);color:#fff;border-radius:18px;padding:26px 26px 22px;margin:0 0 18px}
.sd-hero .t{font-size:2rem;font-weight:800;line-height:1.15;letter-spacing:-.02em}
.sd-hero .s{opacity:.92;margin-top:8px;font-size:1.02rem;line-height:1.5}
.sd-hero .b{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.sd-badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;font-weight:600;background:rgba(255,255,255,.2);color:#fff}
.sd-card{border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.07);border-radius:14px;padding:14px 18px;margin:0 0 12px}
.sd-card .h{font-weight:700;margin-bottom:6px}
.sd-kpi{display:flex;justify-content:space-between;align-items:baseline;gap:12px;border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.07);border-radius:12px;padding:12px 16px;margin:0 0 8px}
.sd-kpi .l{opacity:.75;font-size:.92rem}
.sd-kpi .v{font-size:1.45rem;font-weight:700}
.sd-verdict{border-radius:16px;padding:20px 22px;margin:4px 0 14px;color:#fff}
.sd-verdict.spam{background:linear-gradient(135deg,#dc2626,#f97316)}
.sd-verdict.ham{background:linear-gradient(135deg,#16a34a,#14b8a6)}
.sd-verdict .k{font-size:.8rem;opacity:.9;letter-spacing:.08em;text-transform:uppercase}
.sd-verdict .w{font-size:2.3rem;font-weight:800;line-height:1.1;margin:2px 0 6px}
.sd-verdict .d{opacity:.95;font-size:.95rem}
.sd-gauge{margin:8px 4px 4px}
.sd-gauge .track{position:relative;height:14px;border-radius:999px;background:linear-gradient(90deg,#22c55e,#eab308 50%,#ef4444)}
.sd-gauge .mark{position:absolute;top:-5px;width:6px;height:24px;border-radius:3px;background:#111827;border:2px solid #fff;transform:translateX(-50%)}
.sd-gauge .thr{position:absolute;top:-4px;height:22px;border-left:2px dashed rgba(128,128,128,.95)}
.sd-gauge .lab{display:flex;justify-content:space-between;font-size:.76rem;opacity:.7;margin-top:8px}
.sd-sig{display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-bottom:1px solid rgba(128,128,128,.2)}
.sd-sig:last-child{border-bottom:none}
.sd-dot{flex:0 0 12px;height:12px;border-radius:50%;margin-top:6px}
.sd-sig .tt{font-weight:600}
.sd-sig .dd{opacity:.75;font-size:.9rem}
.sd-bar{display:grid;grid-template-columns:minmax(96px,34%) 1fr 64px;gap:10px;align-items:center;margin:7px 0}
.sd-bar .n{font-size:.92rem;overflow-wrap:anywhere}
.sd-bar .tr{height:16px;border-radius:8px;background:rgba(128,128,128,.18);overflow:hidden}
.sd-bar .f{height:100%;border-radius:8px}
.sd-bar .vl{text-align:right;font-variant-numeric:tabular-nums;font-size:.9rem;font-weight:600}
.sd-donut{display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.sd-ring{position:relative;width:140px;height:140px;flex:0 0 140px}
.sd-ring .c{position:absolute;top:0;left:0;right:0;bottom:0;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 60%,#000 61%);mask:radial-gradient(farthest-side,transparent 60%,#000 61%)}
.sd-ring .m{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:800;font-size:1.5rem}
.sd-ring .m small{font-weight:500;font-size:.72rem;opacity:.7}
.sd-leg{display:flex;align-items:center;gap:8px;margin:5px 0}
.sd-sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.sd-cm{border-collapse:separate;border-spacing:6px;width:100%;table-layout:fixed}
.sd-cm th{font-size:.8rem;font-weight:600;opacity:.75;padding:4px}
.sd-cm td{text-align:center;border-radius:10px;padding:14px 6px}
.sd-cm .v{font-size:1.6rem;font-weight:800}
.sd-cm .p{font-size:.78rem;opacity:.8}
.sd-step{display:flex;gap:14px;padding:9px 0}
.sd-num{flex:0 0 32px;height:32px;border-radius:50%;background:#6366f1;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}
.sd-step .tt{font-weight:600}
.sd-step .dd{opacity:.75;font-size:.92rem}
.sd-hl{line-height:2.1;font-size:1.02rem}
.sd-hl span{border-radius:4px;padding:1px 3px}
.sd-note{font-size:.85rem;opacity:.7}
"""


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def style_tag() -> str:
    """CSS dùng chung cho mọi thành phần; gọi st.markdown(style_tag(), unsafe_allow_html=True) một lần."""
    return "<style>" + re.sub(r"\s+", " ", _CSS).strip() + "</style>"


def hero(title: str, subtitle: str, badges=()) -> str:
    chips = "".join(f'<span class="sd-badge">{_e(b)}</span>' for b in badges)
    return (f'<div class="sd-hero"><div class="t">{_e(title)}</div><div class="s">{_e(subtitle)}</div>'
            f'<div class="b">{chips}</div></div>')


def card(title: str, inner_html: str) -> str:
    """Khung thẻ có tiêu đề; inner_html phải là HTML một dòng do các hàm trong module này tạo ra."""
    return f'<div class="sd-card"><div class="h">{_e(title)}</div>{inner_html}</div>'


def kpi_list(items) -> str:
    """items: [(nhãn, giá trị)] -> các thẻ xếp dọc, mỗi thẻ một chỉ số."""
    return "".join(f'<div class="sd-kpi"><span class="l">{_e(l)}</span><span class="v">{_e(v)}</span></div>'
                   for l, v in items)


def risk_level(spam_prob_pct: float) -> tuple[str, str]:
    """(mô tả, màu) theo xác suất spam (%)."""
    if spam_prob_pct < 25:
        return "Rủi ro thấp", HAM_COLOR
    if spam_prob_pct < 50:
        return "Rủi ro trung bình", "#eab308"
    if spam_prob_pct < 75:
        return "Rủi ro cao", "#f97316"
    return "Rủi ro rất cao", SPAM_COLOR


def verdict_card(label: str, confidence: float, model_label: str, elapsed_ms: float) -> str:
    is_spam = label == "Spam"
    word = "🚨 SPAM" if is_spam else "✅ HAM — email thường"
    detail = f"Độ tin cậy {confidence:.1f}% · Mô hình: {model_label} · {elapsed_ms:.0f} ms"
    return (f'<div class="sd-verdict {"spam" if is_spam else "ham"}"><div class="k">Kết luận</div>'
            f'<div class="w">{_e(word)}</div><div class="d">{_e(detail)}</div></div>')


def gauge(spam_prob_pct: float, threshold: float) -> str:
    """Thanh đo xác suất spam: vạch đen = xác suất hiện tại, vạch nét đứt = ngưỡng phân loại."""
    p = max(0.0, min(100.0, float(spam_prob_pct)))
    t = max(0.0, min(100.0, float(threshold) * 100))
    text, color = risk_level(p)
    return (f'<div class="sd-card"><div class="h">Xác suất spam: {p:.1f}% '
            f'<span style="color:{color}">· {_e(text)}</span></div>'
            f'<div class="sd-gauge"><div class="track"><div class="thr" style="left:{t:.1f}%"></div>'
            f'<div class="mark" style="left:{p:.1f}%"></div></div>'
            f'<div class="lab"><span>0% (Ham)</span><span>ngưỡng {t:.0f}% (nét đứt)</span><span>100% (Spam)</span></div>'
            f'</div></div>')


def signals_html(signals: list[dict]) -> str:
    if not signals:
        rows = (f'<div class="sd-sig"><span class="sd-dot" style="background:{HAM_COLOR}"></span><div>'
                f'<div class="tt">Không phát hiện dấu hiệu đáng ngờ rõ ràng</div>'
                f'<div class="dd">Không có link rút gọn, từ khẩn cấp, nhắc đến tiền thưởng hay yêu cầu tài khoản</div>'
                f'</div></div>')
    else:
        rows = "".join(
            f'<div class="sd-sig"><span class="sd-dot" style="background:{SEVERITY_COLORS[s["severity"]]}"></span>'
            f'<div><div class="tt">{_e(s["title"])} <span class="sd-note">· {SEVERITY_LABELS[s["severity"]]}</span></div>'
            f'<div class="dd">{_e(s["detail"])}</div></div></div>' for s in signals)
    return (f'<div class="sd-card"><div class="h">Dấu hiệu đáng ngờ (theo luật)</div>{rows}'
            f'<div class="sd-note" style="margin-top:6px">Chỉ mang tính tham khảo, độc lập với mô hình ML.</div></div>')


def hbar_chart(items, max_value: float | None = None, fmt: str = "{:.1f}%", default_color: str = ACCENT) -> str:
    """Biểu đồ cột NGANG: nhãn ở bên trái (đọc ngang), thanh ở giữa, giá trị bên phải.

    items: [(nhãn, giá trị)] hoặc [(nhãn, giá trị, màu)].
    """
    items = [tuple(i) + ((default_color,) if len(i) == 2 else ()) for i in items]
    if not items:
        return ""
    top = max_value if max_value else max((abs(v) for _, v, _ in items), default=1) or 1
    rows = []
    for name, value, color in items:
        width = max(0.0, min(100.0, abs(value) / top * 100))
        rows.append(f'<div class="sd-bar"><div class="n">{_e(name)}</div>'
                    f'<div class="tr"><div class="f" style="width:{width:.1f}%;background:{color}"></div></div>'
                    f'<div class="vl">{_e(fmt.format(value))}</div></div>')
    return "".join(rows)


def donut(spam: int, ham: int) -> str:
    total = spam + ham
    pct = (spam / total * 100) if total else 0.0
    ring = (f"conic-gradient({SPAM_COLOR} 0 {pct:.2f}%, {HAM_COLOR} {pct:.2f}% 100%)" if total
            else "conic-gradient(rgba(128,128,128,.3) 0 100%)")
    return (f'<div class="sd-donut"><div class="sd-ring"><div class="c" style="background:{ring}"></div>'
            f'<div class="m">{pct:.0f}%<small>là spam</small></div></div><div>'
            f'<div class="sd-leg"><span class="sd-sw" style="background:{SPAM_COLOR}"></span>Spam: <b>{spam:,}</b></div>'
            f'<div class="sd-leg"><span class="sd-sw" style="background:{HAM_COLOR}"></span>Ham: <b>{ham:,}</b></div>'
            f'<div class="sd-leg"><span class="sd-sw" style="background:rgba(128,128,128,.5)"></span>Tổng: <b>{total:,}</b></div>'
            f'</div></div>')


def confusion_matrix_html(cm: dict) -> str:
    tn, fp, fn, tp = (int(cm[k]) for k in ("tn", "fp", "fn", "tp"))
    good, bad = "rgba(34,197,94,.22)", "rgba(239,68,68,.22)"

    def cell(value, row_total, caption, bg):
        pct = f"{value / row_total * 100:.1f}% của hàng" if row_total else "—"
        return (f'<td style="background:{bg}"><div class="v">{value:,}</div><div class="p">{caption}</div>'
                f'<div class="p">{pct}</div></td>')

    ham_total, spam_total = tn + fp, fn + tp
    return (f'<table class="sd-cm"><tr><th></th><th>Dự đoán: Ham</th><th>Dự đoán: Spam</th></tr>'
            f'<tr><th>Thực tế: Ham</th>{cell(tn, ham_total, "Ham được giữ lại", good)}'
            f'{cell(fp, ham_total, "Báo nhầm thành spam", bad)}</tr>'
            f'<tr><th>Thực tế: Spam</th>{cell(fn, spam_total, "Bỏ sót spam", bad)}'
            f'{cell(tp, spam_total, "Bắt đúng spam", good)}</tr></table>')


def step_list(steps) -> str:
    """steps: [(tiêu đề, mô tả)] -> các bước đánh số xếp dọc."""
    return "".join(f'<div class="sd-step"><div class="sd-num">{i}</div><div><div class="tt">{_e(t)}</div>'
                   f'<div class="dd">{_e(d)}</div></div></div>' for i, (t, d) in enumerate(steps, 1))


def highlight_html(tokens: list[dict]) -> str:
    """Tô đỏ từ đẩy về spam, tô xanh từ đẩy về ham (độ đậm theo mức ảnh hưởng)."""
    parts = []
    for item in tokens:
        impact, safe = item["impact"], _e(item["token"])
        if abs(impact) < 0.01:
            parts.append(safe)
            continue
        alpha = min(0.7, 0.15 + abs(impact) * 2)
        color = "220,53,69" if impact > 0 else "40,167,69"
        tip = _e(f"{impact * 100:+.1f} điểm % xác suất spam")
        parts.append(f'<span title="{tip}" style="background:rgba({color},{alpha:.2f})">{safe}</span>')
    return f'<div class="sd-hl">{" ".join(parts)}</div>'
