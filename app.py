"""
app.py
======
Flask backend cho YH Fin Monitor, thay thế Netlify Functions (không hỗ trợ Python)
để triển khai trên Render.

Luồng xử lý:
1. Phục vụ file tĩnh trong "public/" (index.html, ...) ở route "/".
2. Nhận POST JSON tại "/api/analyze": {"tickers": [...], "period": "...", "interval": "..."}
3. Với mỗi mã, gọi run_stock_analysis_skill() (dev/stock_analysis_script.py).
4. Ghi output_dir/session_file vào thư mục tạm (Render cho phép ghi vào ổ đĩa của instance,
   nhưng dữ liệu không bền vững giữa các lần deploy nên vẫn dùng /tmp để an toàn).
5. Đọc ảnh biểu đồ đã render, encode base64 để trả trực tiếp cho trình duyệt.
6. Trả JSON gồm khuyến nghị (đã lược bỏ DataFrame/NaN không serialize được) + ảnh + mindmap.
"""

import base64
import math
import os
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.join(HERE, "dev")
PUBLIC_DIR = os.path.join(HERE, "public")

if DEV_DIR not in sys.path:
    sys.path.insert(0, DEV_DIR)

from stock_analysis_script import run_stock_analysis_skill  # noqa: E402

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")

MAX_TICKERS = 5  # Giới hạn số mã/lần gọi để tránh timeout request quá lâu.


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


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}

    tickers = body.get("tickers") or []
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]

    if not tickers:
        return jsonify({"error": "Vui lòng nhập ít nhất một mã cổ phiếu."}), 400

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

    return jsonify({"results": results})


@app.get("/")
@app.get("/<path:path>")
def serve_static(path="index.html"):
    return send_from_directory(PUBLIC_DIR, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
