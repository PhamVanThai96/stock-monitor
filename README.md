# stock-monitor

YH Fin Monitor — ứng dụng phân tích kỹ thuật cổ phiếu.

- Backend: Flask (`app.py`) dùng logic trong `dev/stock_analysis_script.py`, deploy trên Render.
- Frontend: file tĩnh trong `public/index.html`, được Flask phục vụ cùng origin (gọi `/api/analyze`).

## Chạy local

```bash
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tới `http://localhost:5000`.

## Deploy lên Render

Dùng cấu hình sẵn trong `render.yaml` (Render Blueprint), hoặc tạo Web Service thủ công với:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`

### Biến môi trường (Telegram, tuỳ chọn)

Nếu dùng tính năng gửi ảnh qua Telegram (`dev/telegram_script.py`), đặt 2 biến môi
trường sau trên Render (Dashboard → Environment) thay vì commit token vào file:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`


