"""
analyze.py
==========
Netlify Function (Python, kiểu handler(event, context) kinh điển) làm cầu nối giữa
trang web tĩnh (public/index.html) và pipeline phân tích kỹ thuật trong
"dev/analysis_script.py".

Luồng xử lý:
1. Nhận POST JSON: {"tickers": ["SSB.VN", "HPG.VN"], "period": "12mo", "interval": "1d"}
2. Với mỗi mã, gọi run_stock_analysis_skill() (đã có sẵn trong analysis_script.py).
3. Ghi output_dir/session_file vào /tmp vì môi trường serverless của Netlify chỉ cho phép
   ghi file tạm ở /tmp (hệ thống file gốc read-only).
4. Đọc ảnh biểu đồ đã render, encode base64 để trả trực tiếp cho trình duyệt (không cần
   lưu trữ file tĩnh lâu dài).
5. Trả JSON gồm khuyến nghị (đã lược bỏ DataFrame không serialize được) + ảnh + mindmap.

Lưu ý: Netlify Functions "modern" (Request/Response) hiện chỉ hỗ trợ chính thức
JS/TS/Go. Python vẫn dùng được qua định dạng legacy handler(event, context) này.
"""

import base64
import json
import math
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# Tìm thư mục dev/ (chứa analysis_script.py) dù được đóng gói theo cấu trúc nào.
_CANDIDATES = [
    os.path.join(HERE, "dev"),
    os.path.join(HERE, "..", "..", "dev"),
    os.path.join(os.getcwd(), "dev"),
    os.path.join(os.getcwd(), "netlify", "functions", "dev"),
]
for _c in _CANDIDATES:
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

from stock_analysis_script import run_stock_analysis_skill  # noqa: E402

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Content-Type": "application/json; charset=utf-8",
}


def _json_safe(value):
    """Chuyển đổi các kiểu numpy/NaN/tuple sang kiểu JSON chuẩn."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def _analyze_one(ticker: str, period: str, interval: str) -> dict:
    tmp_output = "/tmp/output"
    tmp_session = "/tmp/agent_session.json"
    os.makedirs(tmp_output, exist_ok=True)

    result = run_stock_analysis_skill(
        ticker,
        period=period,
        interval=interval,
        output_dir=tmp_output,
        session_file=tmp_session,
    )

    chart_b64 = None
    chart_path = result.get("chart_path")
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "ticker": ticker,
        "ok": True,
        "recommendation": _json_safe(result.get("recommendation", {})),
        "mindmap": result.get("mindmap"),
        "chart_image_base64": chart_b64,
    }


def handler(event, context):
    method = (event.get("httpMethod") or "GET").upper()

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if method != "POST":
        return {
            "statusCode": 405,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Chỉ hỗ trợ phương thức POST."}),
        }

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Body JSON không hợp lệ."}),
        }

    tickers = body.get("tickers") or []
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]

    if not tickers:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Vui lòng nhập ít nhất một mã cổ phiếu."}),
        }

    # Giới hạn số lượng mã / lần gọi để tránh vượt timeout của Netlify Function (mặc định 10s,
    # tối đa 26s trên gói trả phí).
    MAX_TICKERS = 5
    tickers = tickers[:MAX_TICKERS]

    period = body.get("period") or "12mo"
    interval = body.get("interval") or "1d"

    results = []
    for tk in tickers:
        try:
            results.append(_analyze_one(tk, period, interval))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "ticker": tk,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    return {
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps({"results": results}, ensure_ascii=False),
    }
