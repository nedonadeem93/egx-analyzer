import os
import matplotlib
matplotlib.use("Agg")

import streamlit as st

st.set_page_config(page_title="محلل البورصة المصرية",
                   page_icon="📈", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import json
import logging

try:
    from zoneinfo import ZoneInfo
    CAIRO = ZoneInfo("Africa/Cairo")
except Exception:
    CAIRO = None

logging.getLogger().setLevel(logging.ERROR)

try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed(username=os.environ.get("TV_USER") or None,
                    password=os.environ.get("TV_PASS") or None)
    HAS_TV = True
except Exception:
    tv = None
    HAS_TV = False

try:
    from tradingview_screener import Query
    HAS_SCANNER = True
except Exception:
    HAS_SCANNER = False

# ============ قايمتك ⭐ ============

DEFAULT_WATCHLIST = ["COMI", "ABUK", "TMGH", "SWDY", "HRHO", "ETEL",
                     "EAST", "ORHD", "ADIB", "KIMA", "EFID", "PHDC"]

WATCH_FILE = "watchlist.json"

def load_watchlist():
    try:
        with open(WATCH_FILE) as f:
            wl = json.load(f)
        if isinstance(wl, list) and wl:
            return [str(s).strip().upper() for s in wl]
    except Exception:
        pass
    return DEFAULT_WATCHLIST.copy()

def save_watchlist(wl):
    wl = sorted(set(str(s).strip().upper() for s in wl if str(s).strip()))
    try:
        with open(WATCH_FILE, "w") as f:
            json.dump(wl, f)
    except Exception:
        pass
    return wl

# ============ فحص جودة البيانات 🕵️ ============

def data_quality_check(df):
    """بيفحص البيانات الخام ويرجع تقرير جودة + يرملها"""
    problems = []
    cleaned = df.copy()

    n_before = len(cleaned)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    n_dupes = n_before - len(cleaned)
    if n_dupes > 0:
        problems.append(f"⚠️ اتشالت {n_dupes} صف مكرر (عيب معروف في المصدر المجاني)")

    bad = cleaned[(cleaned["Close"] <= 0) |
                  cleaned[["Open", "High", "Low", "Close"]].isna().any(axis=1)]
    if len(bad) > 0:
        problems.append(f"⚠️ اتشالت {len(bad)} صف فيه سعر صفر أو ناقص")
        cleaned = cleaned.drop(bad.index)

    # 🔁 إصلاح: شمعة اليوم الفاضية قبل افتتاح السوق
    # (سعرها = إغلاق أمس وكمياتها صفر — بتلعب على التغير والمؤشرات)
    if len(cleaned) > 1:
        last_vol = cleaned["Volume"].iloc[-1]
        if pd.isna(last_vol) or float(last_vol) <= 0:
            cleaned = cleaned.iloc[:-1]
            problems.append("ℹ️ اتشالت شمعة يوم فاضي — غالباً السوق لسه ما افتحش "
                            "(الحسابات على آخر يوم تداول حقيقي)")

    cleaned = cleaned.sort_index()

    return cleaned, problems

def last_closes_str(df, n=5):
    vals = [f"{v:.2f}" for v in df["Close"].tail(n)]
    return " → ".join(vals)

# ============ 1) جلب البيانات ============

def flat_cols(t):
    cols = t.columns
    if isinstance(cols, pd.MultiIndex):
        cols = cols.get_level_values(0)
        t.columns = cols
    return t

def norm_tv(df):
    df = df.rename(columns={"open": "Open", "high": "High",
                            "low": "Low", "close": "Close",
                            "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    try:
        df.index = df.index.tz_convert(CAIRO)
    except Exception:
        pass
    return df.dropna()

def get_data_yf(symbol):
    df = yf.download(f"{symbol}.CA", period="1y", interval="1d",
                     progress=False, auto_adjust=True)
    df = flat_cols(df).dropna()
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c] * 1000
    return df

def get_hist(symbol):
    if HAS_TV:
        try:
            df = tv.get_hist(symbol, "EGX", Interval.in_daily, 300)
            if df is not None and len(df) > 60:
                return norm_tv(df), "TradingView"
        except Exception:
            pass
    return get_data_yf(symbol), "Yahoo"

def now_cairo():
    if CAIRO:
        return datetime.datetime.now(CAIRO)
    return datetime.datetime.utcnow()

_HIST_CACHE = {}

def get_hist_cached(symbol):
    key = symbol.strip().upper()
    today = now_cairo().date()
    c = _HIST_CACHE.get(key)
    if c and c["day"] == today:
        return c["df"], c["src"]
    try:
        df, src = get_hist(symbol)
    except Exception:
        return None, None
    if df is None or len(df) < 60:
        return None, None
    _HIST_CACHE[key] = dict(day=today, df=df, src=src)
    return df, src

def patch_live(df, price):
    if df is None or df.empty:
        return df
    today = now_cairo().date()
    price = float(price)
    try:
        last_d = df.index[-1].date()
    except Exception:
        return df
    if last_d == today:
        i = df.index[-1]
        df.loc[i, "Close"] = price
        df.loc[i, "High"] = max(price, float(df.loc[i, "High"]))
        df.loc[i, "Low"] = min(price, float(df.loc[i, "Low"]))
    else:
        ts = pd.Timestamp(today)
        if df.index.tz is not None:
            ts = ts.tz_localize(df.index.tz)
        df.loc[ts] = dict(Open=price, High=price,
                          Low=price, Close=price, Volume=np.nan)
        df = df.sort_index()
    return df

def fresh_tv(symbol, last_d):
    if not HAS_TV:
        return None, None
    try:
        t = tv.get_hist(symbol, "EGX", Interval.in_15_minute, 30)
        if t is None or t.empty:
            return None, None
        t = norm_tv(t)
        if t.index[-1].date() >= last_d:
            return float(t["Close"].iloc[-1]), t.index[-1]
    except Exception:
        pass
    return None, None

def fresh_yf(symbol, last_d):
    for iv, per in [("15m", "5d"), ("30m", "5d"), ("1h", "5d")]:
        try:
            t = yf.download(f"{symbol}.CA", period=per, interval=iv,
                            progress=False, auto_adjust=True)
            t = flat_cols(t).dropna()
            if t.empty:
                continue
            ts = t.index[-1]
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            if CAIRO:
                ts = ts.tz_convert(CAIRO)
            if ts.date() >= last_d:
                return float(t["Close"].iloc[-1]) * 1000, ts
        except Exception:
            continue
    return None, None

def get_fresh_price(symbol, last_d):
    px, ts = fresh_tv(symbol, last_d)
    if px is not None:
        return px, ts, "TradingView"
    px, ts = fresh_yf(symbol, last_d)
    if px is not None:
        return px, ts, "Yahoo"
    return None, None, None

# ============ 2) المؤشرات ============

def add_indicators(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    sma20 = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_L"] = sma20 - 2 * std
    return df

# ============ 3) الدعم والمقاومة ============

def find_levels(df, window=8, merge_pct=0.02, max_levels=4):
    highs, lows = [], []
    n = len(df)
    for i in range(window, n - window):
        w = df.iloc[i - window: i + window + 1]
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        if h == w["High"].max():
            highs.append(h)
        if l == w["Low"].min():
            lows.append(l)

    last_close = float(df["Close"].iloc[-1])
    lows = [l for l in lows if l >= last_close * 0.80]
    highs = [h for h in highs if h <= last_close * 1.20]

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        groups = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - groups[-1][-1]) / groups[-1][-1] <= merge_pct:
                groups[-1].append(lvl)
            else:
                groups.append([lvl])
        groups.sort(key=len, reverse=True)
        return [round(float(np.mean(g)), 2) for g in groups[:max_levels]]

    return cluster(lows), cluster(highs)

# ============ 4) السيولة ============

def fmt_num(x):
    x = float(x)
    if x >= 1e9:
        return f"{x/1e9:.2f} مليار"
    if x >= 1e6:
        return f"{x/1e6:.1f} مليون"
    if x >= 1e3:
        return f"{x/1e3:.1f} ألف"
    return f"{x:,.0f}"

def liquidity_info(df):
    last = df.iloc[-1]
    val = df["Close"] * df["Volume"]
    v20 = val.rolling(20).mean().dropna()
    avg_value = float(v20.iloc[-1]) if len(v20) else 0.0
    vol20 = df["Volume"].rolling(20).mean().dropna()
    avg_vol = float(vol20.iloc[-1]) if len(vol20) else 0.0
    today_value = last["Close"] * last["Volume"]
    if pd.isna(today_value):
        today_value = 0.0
    today_value = float(today_value)

    if avg_value >= 100e6:
        rate = "🟢 ممتازة"
    elif avg_value >= 25e6:
        rate = "🟢 جيدة"
    elif avg_value >= 5e6:
        rate = "🟡 متوسطة"
    elif avg_value > 0:
        rate = "🔴 ضعيفة"
    else:
        rate = "⚪ مفيش بيانات"

    rel = today_value / avg_value if avg_value > 0 else 0
    return dict(avg_vol=avg_vol, avg_value=avg_value,
                today_value=today_value, rate=rate, rel=rel)

# ============ 5) القرار ============

def decide(df, price, supports, resistances, liq):
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi = float(last["RSI"])
    score, reasons = 0, []

    if rsi < 30:
        score += 2
        reasons.append("✅ RSI تحت 30 (تشبع بيعي)")
    elif rsi > 70:
        score -= 2
        reasons.append("⚠️ RSI فوق 70 (تشبع شرائي)")

    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        score += 2
        reasons.append("✅ تقاطع MACD صاعد")

    if price > last["SMA50"] > last["SMA200"]:
        score += 1
        reasons.append("✅ الاتجاه العام صاعد")
    elif price < last["SMA50"] < last["SMA200"]:
        score -= 1
        reasons.append("⚠️ الاتجاه العام هابط")

    below = [s for s in supports if s < price]
    above = [r for r in resistances if r > price]
    if below:
        support = max(below)
        real_support = True
    else:
        support = round(price * 0.95, 2)
        real_support = False
    resistance = min(above) if above else round(price * 1.05, 2)

    near_support = (price - support) / price <= 0.02
    if near_support:
        score += 2
        reasons.append(f"✅ السعر ملاصق لدعم {support}")

    if prev["Close"] <= prev["BB_L"] and price > last["BB_L"]:
        score += 1
        reasons.append("✅ ارتداد من البولينجر السفلي")

    if liq["rel"] >= 2:
        reasons.append(f"🔥 تداول النهاردة {liq['rel']:.1f}× المتوسط")
    if 0 < liq["avg_value"] < 5e6:
        reasons.append("⚠️ سيولة ضعيفة — التنفيذ صعب")

    if near_support:
        entry = (round(support * 1.001, 2), round(price, 2))
        entry_note = "عند الدعم — دخول مناسب دلوقتي"
        ref_price = price
    else:
        entry = (round(support, 2), round(support * 1.02, 2))
        if real_support:
            entry_note = f"استنى السعر ينزل لمنطقة {entry[0]} - {entry[1]}"
        else:
            entry_note = (f"مفيش دعم واضح قريب — المنطقة {entry[0]} - {entry[1]} "
                          "محسوبة تحته، والأفضل تستنى دعم حقيقي يتكوّن")
        ref_price = entry[1]

    stop_loss = round(support * 0.97, 2)
    rr = round((resistance - ref_price) / max(ref_price - stop_loss, 0.01), 2)

    if score >= 4:
        decision = "🟢 شراء"
    elif score >= 2:
        decision = "🟡 شراء محتمل"
    elif score <= -2:
        decision = "🔴 بيع / ابتعاد"
    else:
        decision = "⚪ محايد — استنى"

    return dict(support=support, resistance=resistance, entry=entry,
                entry_note=entry_note, stop_loss=stop_loss, rr=rr,
                decision=decision, score=score, reasons=reasons,
                rsi=round(rsi, 1))

def analyze(symbol):
    symbol = symbol.strip().upper()
    df_raw, src = get_hist_cached(symbol)
    if df_raw is None:
        raise ValueError("مفيش بيانات — اتأكد من الرمز")

    df, quality_notes = data_quality_check(df_raw)

    last_d = df.index[-1].date()
    live_px, live_ts, live_src = get_fresh_price(symbol, last_d)

    # 🔁 إصلاح: متعدّلش شمعة جديدة غير لو فيه تداول فعلي النهاردة
    if live_px is not None and live_ts.date() == now_cairo().date():
        price, psrc = live_px, live_src
        p_time = live_ts.strftime("%m-%d %H:%M")
        data_date = live_ts.date()
        df = patch_live(df.copy(), price)
    else:
        price, psrc = float(df["Close"].iloc[-1]), src
        p_time = last_d.strftime("%Y-%m-%d")
        data_date = last_d
        df = df.copy()

    df = add_indicators(df)
    supports, resistances = find_levels(df)
    liq = liquidity_info(df)
    d = decide(df, price, supports, resistances, liq)

    days_stale = (now_cairo().date() - data_date).days

    return dict(symbol=symbol, price=price, psrc=psrc, p_time=p_time,
                stale=data_date != now_cairo().date(),
                days_stale=days_stale,
                last_d=str(last_d),
                quality_notes=quality_notes,
                closes_preview=last_closes_str(df),
                rsi=d["rsi"], support=d["support"],
                resistance=d["resistance"], decision=d["decision"],
                entry=d["entry"], entry_note=d["entry_note"],
                stop_loss=d["stop_loss"], target=d["resistance"],
                rr=d["rr"], reasons=d["reasons"], score=d["score"],
                df=df, liq=liq, supports=supports, resistances=resistances)

# ============ 6) سكانر السوق ============

SYMBOL_DESC = {}
ALL_STOCKS = []

def scan_market():
    global SYMBOL_DESC
    q = (Query().set_markets("egypt")
         .select("name", "description", "close", "change", "volume", "type"))
    # 🔁 إصلاح: رفع حد الـ 50 سهم الافتراضي عشان نجيب السوق كله
    try:
        q = q.limit(500)
    except Exception:
        pass
    _, df = q.get_scanner_data()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    if "type" in df.columns:
        df = df[df["type"] == "stock"]
    if "close" not in df.columns:
        return pd.DataFrame()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.upper().str.strip()
        df = df[df["name"] != ""]
        if "description" in df.columns:
            SYMBOL_DESC.update(dict(zip(df["name"],
                                        df["description"].astype(str))))
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    if "change" not in df.columns:
        df["change"] = 0.0
    return df.rename(columns={"name": "sym"})[["sym", "close", "change", "volume"]]

FALLBACK_STOCKS = ["COMI", "ABUK", "HRHO", "ETEL", "TMGH", "SWDY",
                   "EAST", "ORHD", "ADIB", "KIMA", "EFID", "PHDC",
                   "AMOC", "SKPC", "SUGR", "ECRC", "CIRA", "CLHO",
                   "EGAL", "ISPH", "OKBA", "TALG"]

def fetch_all_symbols():
    global ALL_STOCKS
    if HAS_SCANNER:
        try:
            df = scan_market()
            if not df.empty:
                syms = sorted(s for s in df["sym"].unique() if s.isalnum())
                if len(syms) > 20:
                    ALL_STOCKS = syms
        except Exception:
            pass
    if ALL_STOCKS:
        return ALL_STOCKS
    return sorted(set(DEFAULT_WATCHLIST + FALLBACK_STOCKS))

def scan_fallback_list(symbols):
    rows = []
    for s in symbols:
        df, _ = get_hist_cached(s)
        if df is None:
            continue
        df, _ = data_quality_check(df)
        try:
            px = float(df["Close"].iloc[-1])
            vol = float(df["Volume"].iloc[-1])
            v20 = (df["Close"] * df["Volume"]).rolling(20).mean().dropna()
            avg20 = float(v20.iloc[-1]) if len(v20) else px * vol
            chg = (px / float(df["Close"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            rows.append(dict(sym=s, close=px, change=chg, volume=vol, value=avg20))
        except Exception:
            continue
    return pd.DataFrame(rows)

def deep_row(sym, price, vol_today):
    try:
        df, src = get_hist_cached(sym)
        if df is None:
            return None
        df, _ = data_quality_check(df)
        # 🔁 إصلاح: مفيش شمعة وهمية قبل الافتتاح — لو مفيش تداول النهاردة
        # بنحسب على آخر يوم حقيقي والتغير يطلع بقيمة أمس مش صفر
        if vol_today and vol_today > 0:
            df = patch_live(df.copy(), price)
        else:
            df = df.copy()
            price = float(df["Close"].iloc[-1])
        df = add_indicators(df)
        supports, resistances = find_levels(df)
        liq = liquidity_info(df)
        d = decide(df, price, supports, resistances, liq)
        try:
            chg = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
        except Exception:
            chg = 0.0
        val = price * (vol_today or 0)
