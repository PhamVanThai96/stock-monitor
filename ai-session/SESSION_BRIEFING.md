# SESSION BRIEFING: HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN ĐỊNH LƯỢNG (YH-FIN-MONITOR)

> **Mã phiên:** `STOCK-AI-SESSION-2026-09-17`  
> **Khung thời gian mặc định:** NGÀY (Daily - `interval: "1d"`, `period: "12mo"`)  
> **Trạng thái hệ thống:** Đã chuẩn hóa AI Skill (`stock-analysis`), đóng gói tài liệu vận hành và cấu trúc dữ liệu phiên.

---

## 1. Trạng Thái Watchlist & Kết Quả Quét Gần Nhất (Top Cổ Phiếu VN30)

Tổng hợp từ `ai-session/agent_session.json` và `ai-session/recommendation_log.jsonl`:

| Mã CK | Giá đóng cửa (VNĐ) | Trạng thái MA200 | RSI (14) | Kênh xu hướng | Hành động | Lý do chính |
|---|---|---|---|---|---|---|
| **STB.VN** | 76.200 | Trên MA200 (Tăng) | 48.5 | Ascending | **THEO DÕI** | Đủ điều kiện kỹ thuật nhưng tỷ lệ R:R (1.0) chưa đạt tối thiểu 1:2 (cần chờ nhịp pullback sâu hơn) |
| **HPG.VN** | 21.200 | Dưới MA200 (Giảm) | 42.1 | Descending | **BÁN** | Giá nằm dưới MA200 - xu hướng dài hạn suy giảm, không tham gia bắt đáy |
| **VIC.VN** | 241.200 | Trên MA200 (Tăng) | 54.3 | Ascending | **THEO DÕI** | Giá chưa điều chỉnh về vùng Hỗ trợ mạnh hoặc cạnh dưới kênh tăng |
| **VCB.VN** | 59.600 | Dưới MA200 (Giảm) | 39.8 | Descending | **BÁN** | Giá nằm dưới MA200 - xu hướng dài hạn suy giảm |
| **VNM.VN** | 60.200 | Dưới MA200 (Giảm) | 41.2 | Descending | **BÁN** | Giá nằm dưới MA200 - xu hướng dài hạn suy giảm |
| **SHB.VN** | 11.600 | Dưới MA200 (Giảm) | 38.0 | Descending | **BÁN** | Giá nằm dưới MA200 - xu hướng dài hạn suy giảm |
| **SSB.VN** | 23.450 | Trên MA200 (Tăng) | 91.1 | Ascending | **THEO DÕI** | RSI=91.1 rơi vào vùng cực kỳ quá mua ($\ge 65$), rủi ro điều chỉnh rất cao |

---

## 2. Danh Mục Tài Liệu Kiến Thức Được Đồng Bộ (RAG Baseline)

1. **Tài liệu nến Nhật chuyên sâu:** `documents/Đồ thị nến Nhật - Steve Nison.pdf`
2. **Hướng dẫn vận hành hệ thống:** `documents/HUONG_DAN_SU_DUNG.md`
3. **Kho dữ liệu 407 biểu đồ thực tế:** `documents/Chart/` (Bao gồm các mẫu hình hộp tích lũy, mũi tên dự báo sóng hồi, các mốc hỗ trợ/kháng cự lịch sử).
4. **Hệ thống Skill:** `.agents/skills/stock-analysis/` và `yh-fin-monitor/skills/`.

---

## 3. Các Tham Số Cấu Hình Mặc Định (`dev/config_analysis.json`)

* `period`: `"12mo"` (Quét dữ liệu lịch sử 12 tháng)
* `interval`: `"1d"` (Khung thời gian Ngày - Bắt buộc)
* `candle_order`: `5` (Mức lọc cực trị địa phương fractals)
* `lookback`: `5` (Số phiên gần nhất quét mô hình nến Nhật)
* `output_dir`: `"output"`
* `session_file`: `"ai-session/agent_session.json"`

---

## 4. Kế Hoạch & Nhiệm Vụ Tiếp Theo (Next Steps)

1. **Bổ sung chỉ báo MFI vào pipeline tự động:** Nâng cấp `chart_analysis_skill.py` để tính thêm chỉ số dòng tiền `MFI(14)` và hiển thị trên đồ thị tương đồng với 407 biểu đồ thực tế trong `documents/Chart/`.
2. **Cơ chế Backtest chiến lược:** Xây dựng script kiểm thử lợi nhuận lịch sử (Backtesting) dựa trên bộ quy tắc MUA/BÁN hiện tại.
3. **Mở rộng đa khung thời gian:** Thêm tính năng phân tích kết hợp Đa khung thời gian (Top-down Analysis: Tuần $\rightarrow$ Ngày).
