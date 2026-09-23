import os
import glob
import json
import requests

# 1. Định nghĩa các đường dẫn cấu hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(BASE_DIR, "dev", "key", "tele-bot-token.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def load_telegram_config():
    """Đọc Token và Chat ID.

    Ưu tiên biến môi trường TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (dùng khi deploy
    lên Render hoặc bất kỳ nền tảng nào khác, tránh commit token vào git). Nếu
    không có biến môi trường, fallback đọc từ file JSON cấu hình (chỉ dùng khi
    chạy local, file này đã được thêm vào .gitignore).
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id

    try:
        with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Giả định file JSON có cấu trúc: {"bot_token": "...", "chat_id": "..."}
            return token or config.get("bot_token"), chat_id or config.get("chat_id")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file cấu hình: {e}")
        return token, chat_id

def send_stock_charts():
    """Quét thư mục output, gửi tất cả ảnh PNG qua Telegram và xóa sau khi gửi"""
    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        print("❌ Không tìm thấy Token hoặc Chat ID hợp lệ.")
        return

    # Đường dẫn API gửi ảnh của Telegram
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    
    # Tìm tất cả các file .png trong thư mục output
    search_path = os.path.join(OUTPUT_DIR, "*.png")
    image_files = glob.glob(search_path)
    
    if not image_files:
        print("ℹ️ Không có file hình ảnh nào trong thư mục output để gửi.")
        return

    print(f"🚀 Tìm thấy {len(image_files)} ảnh cần gửi...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"🔄 Đang gửi ảnh: {filename}...")
        
        try:
            # Mở file ảnh ở chế độ binary và gửi qua API
            with open(img_path, 'rb') as img_file:
                payload = {
                    'chat_id': chat_id,
                    'caption': f"📊 Kết quả phân tích chứng khoán: {filename}"  # Nội dung đi kèm ảnh
                }
                files = {
                    'photo': img_file
                }
                
                response = requests.post(url, data=payload, files=files)
                result = response.json()
                
            # Kiểm tra Telegram phản hồi thành công hay thất bại
            if response.status_code == 200 and result.get("ok"):
                print(f"✅ Đã gửi thành công {filename}. Tiến hành xóa file...")
                # Xóa file ảnh sau khi gửi thành công để tránh tràn bộ nhớ
                os.remove(img_path)
            else:
                print(f"❌ Gửi ảnh {filename} thất bại. Lỗi từ Telegram: {result.get('description')}")
                
        except Exception as e:
            print(f"❌ Gửi ảnh {filename} thất bại do lỗi hệ thống: {e}")

if __name__ == "__main__":
    send_stock_charts()
