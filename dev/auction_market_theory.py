#! /usr/bin/env python3

"""
auction_market_theory.py
=========================
Module tích hợp bộ ba công cụ theo Thuyết Đấu Giá (Auction Market Theory - AMT):
  1. Session Volume Profile (SVP): POC / VAH / VAL / HVN / LVN.
  2. TPO Profile (Market Profile): vùng giá trị theo thời gian, phát hiện Single Prints.
  3. Daily VWAP + dải độ lệch chuẩn ±1σ/±2σ/±3σ (reset theo từng phiên giao dịch).

Kèm theo bộ nhận diện hình dạng phân phối (D-Shape / P-Shape / b-Shape) bằng
**weighted moments** (skewness/kurtosis có trọng số khối lượng, tính trực tiếp -
KHÔNG dùng Monte Carlo resampling để đảm bảo kết quả xác định/deterministic),
và bộ sinh tín hiệu giao dịch tự động cho 2 kịch bản của Auction Market Theory:
  - Kịch bản A: Thị trường Cân bằng (D-Shape) -> Chiến lược Mean Reversion.
  - Kịch bản B: Thị trường Mất cân bằng (P/b-Shape) -> Gom/Xả & Breakout.

Tham chiếu lý thuyết: xem file
`/home/worker/WORKs/yh-fin-monitor/auction-market-theory-on-stock-analysis.md`
(đã được review/chính lý bởi chuyên gia phân tích tài chính định lượng).

LƯU Ý QUAN TRỌNG VỀ GIỚI HẠN:
- TPO Profile và Session Volume Profile (SVP) áp dụng được cho MỌI khung thời
  gian (1d, 1wk, hoặc intraday như 15m). Với khung không-intraday (vd "1d"),
  mỗi NẾN được tính là 1 giai đoạn/"chữ cái" TPO (thay vì mỗi `tpo_bar_minutes`
  phút) - tương đương một "Daily/Weekly Range Profile" theo số phiên/nến đã
  chạm qua từng mức giá, khác với TPO intraday truyền thống nhưng vẫn là cách
  tiếp cận hợp lệ của Market Profile khi áp lên dữ liệu đa phiên.
- Daily VWAP + dải ±1σ/±2σ/±3σ CHỈ có ý nghĩa thống kê với dữ liệu intraday
  (vd. interval="15m"), vì VWAP được RESET về 0 vào đầu mỗi phiên giao dịch.
  Với dữ liệu daily (interval="1d"), mỗi phiên chỉ có 1 nến nên "VWAP phiên"
  suy biến về giá đóng cửa - không mang ý nghĩa gì thêm. Hàm liên quan sẽ tự
  động bỏ qua và trả cảnh báo rõ ràng trong `notes` thay vì tính ra số liệu
  vô nghĩa.
- Tín hiệu vào lệnh tự động (`entry_signal`, Kịch bản A/B) vẫn cần ĐỦ cả Volume
  Profile lẫn VWAP+Bands (nên chỉ được sinh ra khi dùng khung intraday); khi
  dùng TPO/Volume Profile trên khung daily/weekly, hệ thống chỉ trả về
  Value Area/POC/HVN/LVN + nhận định cấu trúc D/P/b-Shape, KHÔNG có entry_signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Các khung thời gian intraday hỗ trợ bởi yfinance mà TPO/VWAP session có ý nghĩa.
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

# Giới hạn "period" tối đa mà Yahoo Finance cho phép tải theo từng interval intraday
# (theo tài liệu yfinance/Yahoo, áp dụng để tránh lỗi rỗng dữ liệu khi period quá dài).
INTRADAY_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1h": "730d",
}

def _period_to_days(period: str) -> float:
  import re
  m = re.match(r"^(\d+)(d|wk|mo|y)$", period.strip())
  if not m:
    return float("inf")
  n, unit = int(m.group(1)), m.group(2)
  factor = {"d":1, "wk":7, "mo":30, "y":365}[unit]
  return n*factor

def resolve_period_for_interval(period: str, interval: str) -> str:
    """
    Giới hạn `period` phù hợp với `interval` để tránh gọi yfinance với tổ hợp
    period/interval mà Yahoo Finance sẽ trả về dữ liệu rỗng (vd period="12mo"
    với interval="15m" sẽ bị Yahoo từ chối vì chỉ cho phép tối đa ~60 ngày).
    """
    max_period = INTRADAY_MAX_PERIOD.get(interval, "60d")
    if _period_to_days(period) <= _period_to_days(max_period):
        return period
    return max_period


def is_intraday_interval(interval: str) -> bool:
    return interval in INTRADAY_INTERVALS


# ---------------------------------------------------------------------------
# 1. Weighted moments (Skewness / Excess Kurtosis) - tính trực tiếp, deterministic
# ---------------------------------------------------------------------------
def _weighted_moments(bin_values: np.ndarray):
    """
    Tính mean, std, skewness, excess kurtosis (định nghĩa Fisher, chuẩn = 0)
    theo trọng số khối lượng của từng tầng giá (bin), KHÔNG dùng resampling
    ngẫu nhiên (khác với cách làm sai trong tài liệu tham khảo gốc).

    Trả về None cho tất cả nếu không đủ dữ liệu.
    """
    bin_values = np.asarray(bin_values, dtype=float)
    total = bin_values.sum()
    if total <= 0:
        return None, None, None, None

    weights = bin_values / total
    idx = np.arange(len(bin_values))
    mean = float(np.sum(idx * weights))
    variance = float(np.sum(weights * (idx - mean) ** 2))
    std = float(np.sqrt(variance)) if variance > 0 else 0.0
    if std == 0:
        return mean, std, 0.0, 0.0

    skew = float(np.sum(weights * ((idx - mean) / std) ** 3))
    excess_kurt = float(np.sum(weights * ((idx - mean) / std) ** 4) - 3.0)
    return mean, std, skew, excess_kurt


def detect_profile_shape(bin_values, skew_threshold: float = 0.5,
                          kurt_threshold: float = 1.0,
                          min_bins_with_data: int = 8) -> dict:
    """
    Phân loại hình dạng phân phối khối lượng/TPO theo Auction Market Theory:
      - D_SHAPE: đối xứng (|skew| nhỏ, |kurt| nhỏ) -> Cân bằng (Sideway).
      - P_SHAPE: lệch trái (skew âm mạnh) -> phình to ở mức giá CAO, đuôi mỏng
        ở mức giá THẤP -> Mất cân bằng Tăng.
      - B_SHAPE: lệch phải (skew dương mạnh) -> phình to ở mức giá THẤP, đuôi
        mỏng ở mức giá CAO -> Mất cân bằng Giảm.
      - TRUNG_GIAN: nằm giữa ngưỡng D và P/b, chưa đủ rõ ràng để kết luận.
      - KHONG_DU_DU_LIEU: không đủ khối lượng/số bin có dữ liệu để tin cậy.

    Ngưỡng skew_threshold/kurt_threshold là điểm khởi đầu tham khảo (theo tài
    liệu AMT), CẦN backtest lại trên dữ liệu thực tế của từng mã/thị trường.
    """
    bin_values = np.asarray(bin_values, dtype=float)
    bins_with_data = int(np.sum(bin_values > 0))
    if bins_with_data < min_bins_with_data:
        return {"shape": "KHONG_DU_DU_LIEU", "skewness": None, "kurtosis": None}

    mean, std, skew, kurt = _weighted_moments(bin_values)
    if skew is None:
        return {"shape": "KHONG_DU_DU_LIEU", "skewness": None, "kurtosis": None}

    if abs(skew) < skew_threshold and abs(kurt) < kurt_threshold:
        shape = "D_SHAPE"
    elif skew <= -skew_threshold:
        shape = "P_SHAPE"
    elif skew >= skew_threshold:
        shape = "B_SHAPE"
    else:
        shape = "TRUNG_GIAN"

    return {"shape": shape, "skewness": skew, "kurtosis": kurt}


# ---------------------------------------------------------------------------
# 2. Session Volume Profile (SVP): POC / VAH / VAL / HVN / LVN
# ---------------------------------------------------------------------------
def compute_volume_profile(df: pd.DataFrame, num_bins: int = 60,
                            value_area_pct: float = 0.70) -> dict:
    """
    Xây dựng Volume Profile từ dữ liệu OHLCV dạng nến (không cần dữ liệu tick):
    khối lượng của mỗi nến được PHÂN BỔ TỈ LỆ theo phần trăm chồng lấp giữa
    khoảng giá [Low, High] của nến đó với từng tầng giá (bin) - đây là cách xấp
    xỉ chuẩn mực hơn so với chỉ gán toàn bộ khối lượng vào bin chứa giá Close
    (vì trong 1 nến, giá đã thực sự đi qua toàn bộ khoảng Low-High).

    Trả về dict gồm: edges (biên các bin), bin_volume, poc, vah, val,
    hvn_levels, lvn_levels, value_area_pct_actual.
    """
    lows = df["Low"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    volumes = df["Volume"].to_numpy(dtype=float)

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    if price_max <= price_min:
        raise ValueError("Khoảng giá không hợp lệ để xây dựng Volume Profile (max <= min).")

    edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_volume = np.zeros(num_bins)

    for lo, hi, vol in zip(lows, highs, volumes):
        if vol <= 0:
            continue
        if hi <= lo:
            # Nến không có biên độ (hi == lo, hiếm gặp) -> gán toàn bộ vào 1 bin gần nhất
            idx = int(np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, num_bins - 1))
            bin_volume[idx] += vol
            continue

        start_idx = int(np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, num_bins - 1))
        end_idx = int(np.clip(np.searchsorted(edges, hi, side="right") - 1, 0, num_bins - 1))
        total_range = hi - lo

        for b in range(start_idx, end_idx + 1):
            bin_lo, bin_hi = edges[b], edges[b + 1]
            overlap = min(hi, bin_hi) - max(lo, bin_lo)
            if overlap > 0:
                bin_volume[b] += vol * (overlap / total_range)

    poc_idx = int(np.argmax(bin_volume))
    poc_price = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)

    total_vol = float(bin_volume.sum())
    left = right = poc_idx
    included_vol = bin_volume[poc_idx]
    # Mở rộng Value Area đối xứng từ POC ra hai biên cho tới khi đạt value_area_pct
    # (thuật toán chuẩn của CBOT Market Profile: luôn chọn bên có khối lượng lớn hơn trước).
    while total_vol > 0 and included_vol / total_vol < value_area_pct and (left > 0 or right < num_bins - 1):
        left_vol = bin_volume[left - 1] if left > 0 else -1.0
        right_vol = bin_volume[right + 1] if right < num_bins - 1 else -1.0
        if right_vol >= left_vol:
            right += 1
            included_vol += bin_volume[right]
        else:
            left -= 1
            included_vol += bin_volume[left]

    vah = float(edges[right + 1])
    val = float(edges[left])

    mean_bin_vol = float(bin_volume.mean())
    std_bin_vol = float(bin_volume.std())
    hvn_idx = [i for i, v in enumerate(bin_volume) if v > mean_bin_vol + std_bin_vol and i != poc_idx]
    lvn_idx = [i for i, v in enumerate(bin_volume) if v < max(mean_bin_vol - std_bin_vol, 0.0)]

    return {
        "edges": edges,
        "bin_volume": bin_volume,
        "poc": poc_price,
        "poc_idx": poc_idx,
        "vah": vah,
        "val": val,
        "hvn_levels": [float((edges[i] + edges[i + 1]) / 2) for i in hvn_idx],
        "lvn_levels": [float((edges[i] + edges[i + 1]) / 2) for i in lvn_idx],
        "value_area_pct_actual": float(included_vol / total_vol) if total_vol > 0 else None,
    }


# ---------------------------------------------------------------------------
# 3. TPO Profile (Market Profile): vùng giá trị theo thời gian + Single Prints
# ---------------------------------------------------------------------------
def compute_tpo_profile(df: pd.DataFrame, tpo_bar_minutes: int = 30,
                         num_bins: int = 60, value_area_pct: float = 0.70) -> dict:
    """
    Xây dựng TPO Profile: mỗi "chữ cái" TPO đại diện cho 1 chu kỳ thời gian cố
    định (mặc định 30 phút). Với mỗi chu kỳ, TẤT CẢ các tầng giá (bin) mà giá đã
    đi qua trong chu kỳ đó được cộng thêm 1 "lượt chạm" (không nhân theo khối
    lượng - đây là điểm khác biệt cốt lõi so với Volume Profile).

    Hàm này áp dụng được cho MỌI khung thời gian (`interval`), không chỉ
    intraday: với dữ liệu daily/weekly, khoảng cách giữa các nến (>= 1 ngày)
    luôn lớn hơn `tpo_bar_minutes` nên mỗi NẾN tự động trở thành 1 "chu kỳ"
    riêng (tương đương một Range Profile theo phiên/nến, không phải TPO
    intraday nhiều chữ cái/phiên truyền thống) - vẫn hợp lệ về mặt thuật toán,
    chỉ khác độ phân giải thời gian.

    Yêu cầu df phải có cột "Date" dạng datetime.
    """
    if "Date" not in df.columns:
        raise ValueError("DataFrame cần cột 'Date' (datetime) để xây dựng TPO Profile.")

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["_tpo_period"] = d["Date"].dt.floor(f"{tpo_bar_minutes}min")

    price_min = float(d["Low"].min())
    price_max = float(d["High"].max())
    if price_max <= price_min:
        raise ValueError("Khoảng giá không hợp lệ để xây dựng TPO Profile (max <= min).")

    edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_count = np.zeros(num_bins)

    periods = d.groupby("_tpo_period")
    for _, grp in periods:
        lo = float(grp["Low"].min())
        hi = float(grp["High"].max())
        start_idx = int(np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, num_bins - 1))
        end_idx = int(np.clip(np.searchsorted(edges, hi, side="right") - 1, 0, num_bins - 1))
        bin_count[start_idx:end_idx + 1] += 1

    poc_idx = int(np.argmax(bin_count))
    total_count = float(bin_count.sum())
    left = right = poc_idx
    included = bin_count[poc_idx]
    while total_count > 0 and included / total_count < value_area_pct and (left > 0 or right < num_bins - 1):
        left_c = bin_count[left - 1] if left > 0 else -1.0
        right_c = bin_count[right + 1] if right < num_bins - 1 else -1.0
        if right_c >= left_c:
            right += 1
            included += bin_count[right]
        else:
            left -= 1
            included += bin_count[left]

    # Single Prints: các bin chỉ được chạm đúng 1 lần trong suốt vùng quét -> dấu hiệu
    # giá quét qua rất nhanh (từ chối giá / breakout mạnh).
    single_print_idx = [i for i, c in enumerate(bin_count) if c == 1]

    return {
        "edges": edges,
        "bin_count": bin_count,
        "tpo_value_area_high": float(edges[right + 1]),
        "tpo_value_area_low": float(edges[left]),
        "tpo_poc": float((edges[poc_idx] + edges[poc_idx + 1]) / 2),
        "single_print_levels": [float((edges[i] + edges[i + 1]) / 2) for i in single_print_idx],
        "num_periods": int(periods.ngroups),
    }


# ---------------------------------------------------------------------------
# 4. Daily VWAP + Dải độ lệch chuẩn (±1σ, ±2σ, ±3σ), reset theo từng phiên
# ---------------------------------------------------------------------------
def compute_session_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Công thức: VWAP = cumsum(TypicalPrice * Volume) / cumsum(Volume), RESET về 0
    vào đầu mỗi phiên giao dịch (theo ngày dương lịch của cột "Date").

    Độ lệch chuẩn (σ) được tính là độ lệch chuẩn có trọng số khối lượng
    (volume-weighted std) của TypicalPrice so với VWAP, tích lũy cùng nhịp với
    VWAP trong phiên - đây là cách làm chuẩn cho "VWAP Standard Deviation Bands"
    (tương tự chỉ báo VWAP Bands phổ biến trên TradingView).

    CHỈ áp dụng có ý nghĩa với dữ liệu intraday (nhiều nến/phiên); gọi hàm này
    với dữ liệu daily sẽ không lỗi nhưng kết quả suy biến (mỗi phiên có đúng 1
    điểm dữ liệu -> VWAP = TypicalPrice, std = 0).
    """
    if "Date" not in df.columns:
        raise ValueError("DataFrame cần cột 'Date' (datetime) để xác định phiên giao dịch.")

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    session_key = d["Date"].dt.date

    typical_price = (d["High"] + d["Low"] + d["Close"]) / 3.0
    volume = d["Volume"].astype(float)
    tp_vol = typical_price * volume

    cum_vol = volume.groupby(session_key).cumsum()
    cum_tp_vol = tp_vol.groupby(session_key).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    dev_sq_vol = ((typical_price - vwap) ** 2) * volume
    cum_dev_sq_vol = dev_sq_vol.groupby(session_key).cumsum()
    variance = cum_dev_sq_vol / cum_vol.replace(0, np.nan)
    std = np.sqrt(variance)

    out = df.copy()
    out["VWAP"] = vwap.values
    for k in (1, 2, 3):
        out[f"VWAP_UP{k}"] = (vwap + k * std).values
        out[f"VWAP_DOWN{k}"] = (vwap - k * std).values
    return out


# ---------------------------------------------------------------------------
# 5. Sinh nhận định & tín hiệu giao dịch tự động theo 2 Kịch bản AMT
# ---------------------------------------------------------------------------
_SHAPE_NOTES = {
    "D_SHAPE": "⚖️ Cấu trúc CÂN BẰNG (D-Shape): thị trường đang đi ngang quanh POC. "
               "Ưu tiên chiến lược Mean Reversion (Kịch bản A), tránh đánh Breakout.",
    "P_SHAPE": "📈 Cấu trúc MẤT CÂN BẰNG TĂNG (P-Shape): khối lượng/thời gian dồn về vùng giá cao, "
               "đuôi mỏng phía dưới (Single Prints). Có thể đang trong giai đoạn Phân phối "
               "(Distribution) ở đỉnh hoặc Breakout tăng đang được chấp nhận (Kịch bản B).",
    "B_SHAPE": "📉 Cấu trúc MẤT CÂN BẰNG GIẢM (b-Shape): khối lượng/thời gian dồn về vùng giá thấp, "
               "đuôi mỏng phía trên. Có thể đang trong giai đoạn Gom hàng (Accumulation) ở đáy "
               "hoặc Breakout giảm đang được chấp nhận (Kịch bản B).",
    "TRUNG_GIAN": "🔍 Cấu trúc chưa rõ ràng (giữa D-Shape và P/b-Shape) - nên chờ thêm dữ liệu, "
                  "hạn chế vào lệnh mới cho tới khi cấu trúc rõ ràng hơn.",
    "KHONG_DU_DU_LIEU": "⚠️ Không đủ khối lượng/số nến quan sát để xác định hình dạng phân phối "
                        "một cách đáng tin cậy (cần backtest thêm với dữ liệu dài hơn).",
}


def _shape_note(shape: str) -> str:
    return _SHAPE_NOTES.get(shape, "Không xác định cấu trúc thị trường.")


def _is_shooting_star(open_, high, low, close, body_ratio: float = 2.0, wick_tolerance: float = 0.5) -> bool:
    body = abs(close - open_)
    upper_wick = high - max(close, open_)
    lower_wick = min(close, open_) - low
    if body == 0:
        body = (high - low) * 0.05 or 1e-9
    return upper_wick > body_ratio * body and lower_wick < wick_tolerance * body


def _is_hammer(open_, high, low, close, body_ratio: float = 2.0, wick_tolerance: float = 0.5) -> bool:
    body = abs(close - open_)
    upper_wick = high - max(close, open_)
    lower_wick = min(close, open_) - low
    if body == 0:
        body = (high - low) * 0.05 or 1e-9
    return lower_wick > body_ratio * body and upper_wick < wick_tolerance * body


def _detect_entry_signal(df_ind: pd.DataFrame, vp: dict, vwap_df: pd.DataFrame, shape: str) -> dict | None:
    """
    Sinh tín hiệu vào lệnh tự động theo 2 kịch bản của Auction Market Theory
    (mục 2 trong prompt-AMT-update.md):
      - Kịch bản A (D-Shape, Mean Reversion): SHORT tại Selling Excess (VAH + VWAP+2σ/+3σ
        + nến Shooting Star); LONG tại Buying Excess (VAL + VWAP-2σ/-3σ + nến Hammer).
      - Kịch bản B (P/b-Shape, Breakout/Trend): xác nhận Breakout khi giá phá VAH/VAL
        kèm khối lượng lớn -> khuyến nghị huỷ Mean Reversion, đánh thuận xu hướng.
    """
    last = df_ind.iloc[-1]
    vwap_last = vwap_df.iloc[-1]

    close = float(last["Close"])
    open_ = float(last["Open"])
    high = float(last["High"])
    low = float(last["Low"])
    vol_ratio = float(last["VOL_RATIO"]) if "VOL_RATIO" in df_ind.columns and pd.notna(last.get("VOL_RATIO")) else 1.0

    vah, val, poc = vp["vah"], vp["val"], vp["poc"]
    vwap_center = vwap_last.get("VWAP")
    up2, up3 = vwap_last.get("VWAP_UP2"), vwap_last.get("VWAP_UP3")
    down2, down3 = vwap_last.get("VWAP_DOWN2"), vwap_last.get("VWAP_DOWN3")

    if shape == "D_SHAPE":
        if (close > vah and pd.notna(up2) and (close > up2 or (pd.notna(up3) and close > up3))
                and _is_shooting_star(open_, high, low, close)):
            return {
                "scenario": "A",
                "action": "SHORT",
                "entry": close,
                "target1": float(vwap_center) if pd.notna(vwap_center) else None,
                "target2": poc,
                "note": (f"⚡ Kịch bản A (D-Shape - Mean Reversion): Thị trường từ chối giá cao "
                         f"(Selling Excess) tại VAH ({vah:,.0f}), vượt dải VWAP +2σ/+3σ, xuất hiện "
                         f"nến rút râu trên (Shooting Star). Kích hoạt lệnh SHORT. "
                         f"Target 1: VWAP ({vwap_center:,.0f} nếu có), Target 2: POC phiên ({poc:,.0f})."),
            }
        if (close < val and pd.notna(down2) and (close < down2 or (pd.notna(down3) and close < down3))
                and _is_hammer(open_, high, low, close)):
            return {
                "scenario": "A",
                "action": "LONG",
                "entry": close,
                "target1": float(vwap_center) if pd.notna(vwap_center) else None,
                "target2": poc,
                "note": (f"⚡ Kịch bản A (D-Shape - Mean Reversion): Thị trường từ chối giá thấp "
                         f"(Buying Excess) tại VAL ({val:,.0f}), vượt dải VWAP -2σ/-3σ, xuất hiện "
                         f"nến rút chân (Hammer). Kích hoạt lệnh LONG. "
                         f"Target 1: VWAP ({vwap_center:,.0f} nếu có), Target 2: POC phiên ({poc:,.0f})."),
            }
        return None

    # Kịch bản B: P_SHAPE / B_SHAPE / TRUNG_GIAN -> kiểm tra breakout được chấp nhận (Accepted Auction)
    if close > vah and vol_ratio >= 1.5:
        return {
            "scenario": "B",
            "action": "LONG_TREND",
            "entry": close,
            "target1": None,
            "target2": None,
            "note": ("⚡ Kịch bản B (Breakout được chấp nhận): giá phá vỡ VAH "
                     f"({vah:,.0f}) kèm khối lượng đột biến (x{vol_ratio:.2f} SMA20). "
                     "Thị trường chuyển sang trạng thái Mất cân bằng (Trend). "
                     "Huỷ toàn bộ lệnh đảo chiều Mean Reversion, ưu tiên đánh thuận xu hướng Long."),
        }
    if close < val and vol_ratio >= 1.5:
        return {
            "scenario": "B",
            "action": "SHORT_TREND",
            "entry": close,
            "target1": None,
            "target2": None,
            "note": ("⚡ Kịch bản B (Breakout được chấp nhận): giá phá vỡ VAL "
                     f"({val:,.0f}) kèm khối lượng đột biến (x{vol_ratio:.2f} SMA20). "
                     "Thị trường chuyển sang trạng thái Mất cân bằng (Trend). "
                     "Huỷ toàn bộ lệnh đảo chiều Mean Reversion, ưu tiên đánh thuận xu hướng Short."),
        }
    return None


def generate_amt_signals(df_ind: pd.DataFrame, interval: str, indicators: list | None = None,
                          tpo_bar_minutes: int = 30, num_bins: int = 60) -> dict:
    """
    Hàm điều phối chính: tính các chỉ báo AMT được yêu cầu trong `indicators`
    (danh sách con của {"tpo", "volume_profile", "vwap"}) và sinh nhận định/
    tín hiệu giao dịch tự động.

    `num_bins` (mặc định 60, tăng từ 24): số tầng giá (bins) chia nhỏ khoảng
    [Low, High] của toàn bộ dữ liệu để dựng TPO/Volume Profile - càng lớn thì
    các cột/khối càng NHỎ và NHIỀU hơn, biểu diễn chi tiết hơn sự tích luỹ
    khối lượng/thời gian theo từng vùng giá hẹp.

    Trả về dict luôn có khoá "notes" (list[str]) và "entry_signal" (dict|None)
    để dễ dàng merge vào `rec["reasons"]` của pipeline khuyến nghị hiện tại.
    """
    indicators = indicators or []
    intraday = is_intraday_interval(interval)

    result = {
        "indicators_requested": list(indicators),
        "interval": interval,
        "shape": None,
        "skewness": None,
        "kurtosis": None,
        "poc": None,
        "vah": None,
        "val": None,
        "hvn_levels": [],
        "lvn_levels": [],
        "tpo": None,
        "vwap_last": None,
        "sigma_bands": None,
        "entry_signal": None,
        "notes": [],
    }

    vp = None
    if "volume_profile" in indicators or "tpo" in indicators:
        # Volume Profile luôn được tính khi TPO được yêu cầu vì cần nó để phát
        # hiện hình dạng D/P/b-Shape dùng chung cho cả 2 chỉ báo.
        vp = compute_volume_profile(df_ind, num_bins=num_bins)
        shape_info = detect_profile_shape(vp["bin_volume"])
        result.update({
            "poc": vp["poc"], "vah": vp["vah"], "val": vp["val"],
            "hvn_levels": vp["hvn_levels"], "lvn_levels": vp["lvn_levels"],
            "shape": shape_info["shape"], "skewness": shape_info["skewness"],
            "kurtosis": shape_info["kurtosis"],
        })
        if "volume_profile" in indicators:
            result["notes"].append(_shape_note(shape_info["shape"]))
            result["volume_profile"] = vp

    if "tpo" in indicators:
        tpo = compute_tpo_profile(df_ind, tpo_bar_minutes=tpo_bar_minutes, num_bins=num_bins)
        result["tpo"] = tpo
        if not intraday:
            result["notes"].append(
                f"ℹ️ TPO Profile đang tính trên khung '{interval}': mỗi NẾN (không phải mỗi "
                f"{tpo_bar_minutes} phút) được tính là 1 giai đoạn thời gian/'chữ cái' TPO. "
                "Đây là TPO theo phiên/nến (Daily/Weekly Range Profile), khác với TPO intraday "
                "truyền thống (nhiều chữ cái trong 1 phiên) nhưng vẫn phản ánh đúng vùng giá trị "
                "(Value Area) theo số phiên/nến đã chạm qua từng mức giá."
            )
        if tpo["single_print_levels"]:
            result["notes"].append(
                f"🟨 TPO phát hiện {len(tpo['single_print_levels'])} vùng Single Prints "
                "(giá quét qua rất nhanh) - các mức này có thể trở thành hỗ trợ/kháng cự mạnh."
            )

    vwap_df = None
    if "vwap" in indicators:
        if not intraday:
            result["notes"].append(
                f"⚠️ Daily VWAP + dải σ chỉ có ý nghĩa với dữ liệu intraday; bỏ qua vì khung hiện tại là '{interval}'."
            )
        else:
            vwap_df = compute_session_vwap_bands(df_ind)
            last = vwap_df.iloc[-1]
            result["vwap_last"] = float(last["VWAP"]) if pd.notna(last["VWAP"]) else None
            sigma_bands = {}
            for k in (1, 2, 3):
                up_val = last.get(f"VWAP_UP{k}")
                down_val = last.get(f"VWAP_DOWN{k}")
                sigma_bands[f"+{k}sigma"] = float(up_val) if pd.notna(up_val) else None
                sigma_bands[f"-{k}sigma"] = float(down_val) if pd.notna(down_val) else None
            result["sigma_bands"] = sigma_bands
            result["vwap_df"] = vwap_df

    # Chỉ sinh tín hiệu vào lệnh cụ thể khi có ĐỦ cả Volume Profile lẫn VWAP+Bands
    # (đúng theo quy trình 3 bước TPO -> Volume Profile -> VWAP trong tài liệu AMT).
    if vp is not None and vwap_df is not None and result["shape"] is not None:
        signal = _detect_entry_signal(df_ind, vp, vwap_df, result["shape"])
        if signal:
            result["entry_signal"] = signal
            result["notes"].append(signal["note"])

    return result


# ---------------------------------------------------------------------------
# 6. Vẽ TPO / Volume Profile / VWAP+Bands lên biểu đồ nến (matplotlib) hiện có
# ---------------------------------------------------------------------------
def render_amt_overlays(ax, n: int, amt: dict, indicators: list) -> dict:
    """
    Vẽ các lớp overlay AMT lên trục `ax` (ax1 - bảng nến chính của
    `plot_advanced_chart` trong stock_analysis_script.py), đúng theo bảng màu/
    kiểu nét quy định trong "prompt-AMT-update.md" mục 1:
      - TPO Profile: khối màu xám (Block/Histogram), LEFT-ALIGNED (bên trái nến đầu tiên).
      - Session Volume Profile (SVP): khối Up/Down Volume, RIGHT-ALIGNED (bên phải nến cuối).
      - Daily VWAP + dải ±1σ/±2σ/±3σ: vẽ dọc theo toàn bộ chuỗi nến.

    `amt` là dict trả về từ `generate_amt_signals()` (cần chứa các khoá
    "volume_profile"/"tpo"/"vwap_df" tương ứng với `indicators` được yêu cầu).

    Trả về {"left_margin": int, "right_margin": int}: số đơn vị trục X cần cộng
    thêm vào `ax.set_xlim(...)` ở nơi gọi để TPO/Volume Profile không bị cắt ảnh.

    LƯU Ý: Volume Profile ở đây được tách màu Up/Down (xanh ngọc/hồng nhạt) dựa
    trên quy ước trực quan "bin nằm trên/dưới POC" (KHÔNG phải phân loại
    Up-tick/Down-tick thực tế vì dữ liệu nguồn chỉ là OHLCV theo nến, không có
    dữ liệu tick khớp lệnh chi tiết) - cần nêu rõ giới hạn này khi diễn giải.
    """
    indicators = indicators or []
    margins = {"left_margin": 0, "right_margin": 0}

    # ---- 1.2 Session Volume Profile (SVP): RIGHT-ALIGNED ----
    vp = amt.get("volume_profile")
    if "volume_profile" in indicators and vp:
        edges = vp["edges"]
        bin_volume = np.asarray(vp["bin_volume"], dtype=float)
        max_vol = float(bin_volume.max()) if bin_volume.max() > 0 else 1.0
        # Bề rộng khối tỉ lệ với số nến hiển thị, giới hạn [8, 22] cột (tăng so với
        # trước để vẫn hiển thị rõ dù num_bins lớn hơn -> cột nhỏ/nhiều hơn).
        vp_width_bars = max(8.0, min(22.0, n * 0.16))

        # Tô nền mờ toàn bộ vùng giá trị 70% khối lượng (Value Area: VAL <-> VAH)
        # trên SUỐT chiều rộng biểu đồ (axhspan dùng toạ độ trục X theo tỉ lệ axes,
        # không bị cắt khi set_xlim thay đổi sau đó) để dễ nhận diện vùng thị
        # trường giao dịch nhiều nhất.
        ax.axhspan(vp["val"], vp["vah"], color="#7e57c2", alpha=0.06, zorder=0.3,
                   label="VA 70% (Volume Profile)")

        base_x = n - 0.5  # mép phải của nến cuối cùng
        mean_vol = float(bin_volume.mean())
        std_vol = float(bin_volume.std())
        for i in range(len(bin_volume)):
            vol = bin_volume[i]
            if vol <= 0:
                continue
            lo, hi = edges[i], edges[i + 1]
            width = (vol / max_vol) * vp_width_bars
            bin_center = (lo + hi) / 2
            # Quy ước màu: bin ở trên POC -> xu hướng Mua (Up Volume, xanh ngọc);
            # bin ở dưới POC -> xu hướng Bán (Down Volume, hồng nhạt).
            color = "#26a69a" if bin_center >= vp["poc"] else "#ef5350"
            ax.barh(bin_center, width, left=base_x, height=(hi - lo) * 0.9,
                    color=color, alpha=0.3, zorder=3, edgecolor="none")
            if vol > mean_vol + std_vol:
                # HVN (High Volume Node): tô thêm lớp xanh lam mờ làm nổi bật "bức tường" thanh khoản.
                ax.barh(bin_center, width, left=base_x, height=(hi - lo) * 0.9,
                        color="#1E88E5", alpha=0.18, zorder=3.1, edgecolor="none")
            # LVN (Low Volume Node): bin < mean - std -> cố tình KHÔNG tô màu (để trống),
            # đúng yêu cầu "hiển thị bằng các khoảng hở không màu".

        ax.axhline(y=vp["poc"], color="purple", linestyle="-", linewidth=1.8, alpha=0.95,
                   zorder=4, label="POC (Volume Profile)")
        ax.axhline(y=vp["vah"], color="#4CAF50", linestyle="--", linewidth=1.2, alpha=0.75,
                   zorder=4, label="VAH (Volume Profile)")
        ax.axhline(y=vp["val"], color="#F44336", linestyle="--", linewidth=1.2, alpha=0.75,
                   zorder=4, label="VAL (Volume Profile)")

        # Nhãn giá trực tiếp cạnh khối Volume Profile (dễ đọc hơn so với chỉ xem
        # chú giải/legend ở góc biểu đồ, nhất là khi có nhiều overlay cùng bật).
        label_x = base_x + vp_width_bars * 1.04
        for price_val, text_prefix, text_color in (
            (vp["poc"], "POC", "purple"), (vp["vah"], "VAH", "#2e7d32"), (vp["val"], "VAL", "#c62828")
        ):
            ax.text(label_x, price_val, f"{text_prefix} {price_val:,.0f}", fontsize=7,
                    color=text_color, fontweight="bold", va="center", ha="left", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8))

        margins["right_margin"] = vp_width_bars + 5  # chừa thêm chỗ cho nhãn giá POC/VAH/VAL

    # ---- 1.1 TPO Profile: LEFT-ALIGNED ----
    tpo = amt.get("tpo")
    if "tpo" in indicators and tpo:
        edges = tpo["edges"]
        bin_count = np.asarray(tpo["bin_count"], dtype=float)
        max_count = float(bin_count.max()) if bin_count.max() > 0 else 1.0
        tpo_width_bars = max(8.0, min(22.0, n * 0.16))
        margins["left_margin"] = tpo_width_bars + 1

        base_x = -0.5  # mép trái của nến đầu tiên
        va_hi, va_lo = tpo["tpo_value_area_high"], tpo["tpo_value_area_low"]
        bin_h = (edges[1] - edges[0]) if len(edges) > 1 else 1.0

        # Tô nền mờ vùng giá trị TPO (Value Area 70% theo thời gian), dùng màu
        # KHÁC với Volume Profile (hổ phách nhạt thay vì tím) để không nhầm lẫn
        # khi cả 2 overlay cùng được bật đồng thời.
        ax.axhspan(va_lo, va_hi, color="#8d6e63", alpha=0.045, zorder=0.2,
                   label="VA 70% (TPO)")

        for i in range(len(bin_count)):
            cnt = bin_count[i]
            if cnt <= 0:
                continue
            lo, hi = edges[i], edges[i + 1]
            width = (cnt / max_count) * tpo_width_bars
            # Mặc định ẩn TPO Letters, chỉ hiển thị dạng khối màu (Block/Histogram).
            in_value_area = lo < va_hi and hi > va_lo
            color = "#4A4A4A" if in_value_area else "#D3D3D3"
            alpha = 0.20 if in_value_area else 0.05
            bin_center = (lo + hi) / 2
            ax.barh(bin_center, width, left=base_x - width, height=bin_h * 0.9,
                    color=color, alpha=alpha, zorder=3, edgecolor="none")

        ax.axhline(y=tpo["tpo_poc"], color="#616161", linestyle="-", linewidth=1.3, alpha=0.7,
                   zorder=3, label="TPO POC")
        ax.axhline(y=va_hi, color="#8d6e63", linestyle=":", linewidth=1.1, alpha=0.65,
                   zorder=3, label="TPO VAH")
        ax.axhline(y=va_lo, color="#8d6e63", linestyle=":", linewidth=1.1, alpha=0.65,
                   zorder=3, label="TPO VAL")

        # Buying/Selling Tails: Single Prints ở biên TRÊN (Selling Tail, tím nhạt)
        # hoặc biên DƯỚI (Buying Tail, vàng cát) của profile - đánh dấu bằng gạch dọc.
        if tpo["single_print_levels"]:
            mid_price = (edges[0] + edges[-1]) / 2
            tick_x = base_x - tpo_width_bars * 0.5
            for level in tpo["single_print_levels"]:
                is_selling_tail = level >= mid_price  # đuôi trên = từ chối giá cao = khung Bán
                color = "#C9A0DC" if is_selling_tail else "#E0C080"  # tím nhạt / vàng cát
                ax.vlines(tick_x, level - bin_h * 0.4, level + bin_h * 0.4,
                           color=color, linewidth=2.2, zorder=4)

        ax.axhline(y=tpo["tpo_poc"], color="#616161", linestyle="-", linewidth=1.0, alpha=0.5, zorder=3)

    # ---- 1.3 Daily VWAP + Dải độ lệch chuẩn ----
    vwap_df = amt.get("vwap_df")
    if "vwap" in indicators and vwap_df is not None:
        x_v = np.arange(len(vwap_df))
        ax.plot(x_v, vwap_df["VWAP"], color="#FF9800", linewidth=2.0, zorder=5, label="VWAP (Phiên)")

        sigma_styles = {
            1: dict(color="gray", linestyle=(0, (1, 1)), linewidth=1.0, alpha=0.8),
            2: dict(color="#EEFF41", linestyle="--", linewidth=1.0, alpha=0.9),
            3: dict(color="#FF1744", linestyle="--", linewidth=1.0, alpha=0.9),
        }
        for k, style in sigma_styles.items():
            up_col, down_col = f"VWAP_UP{k}", f"VWAP_DOWN{k}"
            if up_col not in vwap_df.columns:
                continue
            label = f"VWAP ±{k}σ" if k in (1, 2, 3) else None
            ax.plot(x_v, vwap_df[up_col], zorder=5, label=label, **style)
            ax.plot(x_v, vwap_df[down_col], zorder=5, **style)

        # Dải 95% xác suất (±2σ): tô nền vàng cực mờ để nhận diện vùng phân phối.
        if "VWAP_UP2" in vwap_df.columns and "VWAP_DOWN2" in vwap_df.columns:
            ax.fill_between(x_v, vwap_df["VWAP_DOWN2"], vwap_df["VWAP_UP2"],
                             color="#EEFF41", alpha=0.03, zorder=1)

        # Cảnh báo "Biên quá đà" (±3σ): chỉ xuất hiện khi nến cuối thực sự chạm dải.
        last_row = vwap_df.iloc[-1]
        last_close = float(last_row["Close"])
        up3 = last_row.get("VWAP_UP3")
        down3 = last_row.get("VWAP_DOWN3")
        if pd.notna(up3) and last_close >= up3:
            ax.annotate("⚠️ Chạm dải VWAP +3σ (quá đà mua)", xy=(x_v[-1], last_close),
                        xytext=(0, 20), textcoords="offset points", fontsize=7.5,
                        fontweight="bold", color="#FF1744", ha="center", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="#FF1744", alpha=0.9))
        elif pd.notna(down3) and last_close <= down3:
            ax.annotate("⚠️ Chạm dải VWAP -3σ (quá đà bán)", xy=(x_v[-1], last_close),
                        xytext=(0, -24), textcoords="offset points", fontsize=7.5,
                        fontweight="bold", color="#FF1744", ha="center", va="top", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="#FF1744", alpha=0.9))

    return margins


if __name__ == "__main__":
    # --- Kiểm thử nhanh bằng dữ liệu giả lập (không cần mạng) ---
    rng = np.random.default_rng(7)
    n = 200
    dates = pd.date_range("2026-01-01 09:15", periods=n, freq="15min")
    prices = 100 + np.cumsum(rng.normal(0, 0.3, size=n))
    highs = prices + rng.uniform(0.1, 0.6, size=n)
    lows = prices - rng.uniform(0.1, 0.6, size=n)
    opens = prices + rng.normal(0, 0.1, size=n)
    closes = prices + rng.normal(0, 0.1, size=n)
    volumes = rng.integers(100, 1000, size=n).astype(float)

    df_test = pd.DataFrame({
        "Date": dates, "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    })
    df_test["VOL_RATIO"] = 1.0

    out = generate_amt_signals(df_test, interval="15m", indicators=["tpo", "volume_profile", "vwap"])
    print("Shape:", out["shape"], "| Skew:", out["skewness"], "| Kurtosis:", out["kurtosis"])
    print("POC/VAH/VAL:", out["poc"], out["vah"], out["val"])
    print("VWAP cuối:", out["vwap_last"], "| Sigma bands:", out["sigma_bands"])
    for note in out["notes"]:
        print(" -", note)
