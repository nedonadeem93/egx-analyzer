import os
import matplotlib
matplotlib.use("Agg")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
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

DATA_DIR = "/data" if os.path.isdir("/data") else "."
WATCH_FILE = os.path.join(DATA_DIR, "watchlist.json")

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

STOCKS = load_watchlist()

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
    support = max(below) if below else round(price * 0.95, 2)
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
    else:
        entry = (round(support, 2), round(support * 1.02, 2))
        entry_note = f"استنى السعر ينزل لمنطقة {entry[0]} - {entry[1]}"

    stop_loss = round(support * 0.97, 2)
    rr = round((resistance - price) / max(price - stop_loss, 0.01), 2)

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
    df, src = get_hist_cached(symbol)
    if df is None:
        raise ValueError("مفيش بيانات — اتأكد من الرمز")

    last_d = df.index[-1].date()
    live_px, live_ts, live_src = get_fresh_price(symbol, last_d)

    if live_px is not None:
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

    return dict(symbol=symbol, price=price, psrc=psrc, p_time=p_time,
                stale=data_date != now_cairo().date(),
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
        df = patch_live(df.copy(), price)
        df = add_indicators(df)
        supports, resistances = find_levels(df)
        liq = liquidity_info(df)
        d = decide(df, price, supports, resistances, liq)
        try:
            chg = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100
        except Exception:
            chg = 0.0
        val = price * (vol_today or 0)
        if val <= 0:
            val = liq["avg_value"]
        return dict(sym=sym, price=price, chg=chg, val=val,
                    score=d["score"], dec=d["decision"], d=d)
    except Exception:
        return None

# ============ 7) جدول النتايج ============

HEADERS = ["السهم", "السعر", "التغير", "سيولة اليوم", "الدرج", "القرار",
           "الدعم", "المقاومة", "منطقة الدخول", "وقف الخسارة", "R/R", "ملاحظات"]

def empty_table():
    return pd.DataFrame(columns=HEADERS)

def build_table(res):
    if res is None or not len(res):
        return empty_table()
    res = res.sort_values(["score", "val"], ascending=False)
    rows = []
    for r in res.itertuples():
        d = r.d
        notes = " • ".join(d["reasons"][:3]) if d["reasons"] else "—"
        rows.append([
            r.sym, f"{r.price:.2f}", f"{r.chg:+.1f}%",
            fmt_num(r.val), f"{r.score:+d}", r.dec,
            f"{d['support']:.2f}", f"{d['resistance']:.2f}",
            f"{d['entry'][0]} – {d['entry'][1]}",
            f"{d['stop_loss']:.2f}", f"{d['rr']:.1f}", notes])
    return pd.DataFrame(rows, columns=HEADERS)

def run_scan(scope, min_liq, top_n):
    try:
        yield "⏳ عم نجيب بيانات السوق...", empty_table(), ""
        warn = ""
        if scope == "كل السوق" and HAS_SCANNER:
            try:
                scan_df = scan_market()
            except Exception:
                scan_df = None
            if scan_df is None or scan_df.empty:
                warn = "⚠️ السكانر رجّع حاجة فاضية — استخدمنا قايمة بديلة محدودة\n\n"
                scan_df = scan_fallback_list(FALLBACK_STOCKS)
        elif scope == "كل السوق":
            warn = "⚠️ مكتبة السكانر مش متاحة — استخدمنا قايمة بديلة محدودة\n\n"
            scan_df = scan_fallback_list(FALLBACK_STOCKS)
        else:
            scan_df = scan_fallback_list(list(STOCKS))

        if scan_df is None or scan_df.empty:
            if scope == "⭐ قايمتي":
                yield ("⚪ قايمتك فاضية أو مفيش منها بيانات — ضيف أسهم من تاب ⭐ قايمتي",
                       empty_table(), "")
            else:
                yield "❌ فشل المسح — جرب تاني", empty_table(), ""
            return

        n_all = len(scan_df)
        if "value" not in scan_df.columns:
            scan_df["value"] = scan_df["close"] * scan_df["volume"]
        scan_df = scan_df[(scan_df["close"] > 0) &
                          (scan_df["value"] >= min_liq * 1e6)]
        scan_df = scan_df.sort_values("value", ascending=False)
        if scope == "كل السوق":
            scan_df = scan_df.head(top_n)
        scan_df = scan_df.reset_index(drop=True)
        if scan_df.empty:
            yield "⚪ مفيش أسهم فوق حد السيولة ده — قلل الحد وجرّب تاني", empty_table(), ""
            return

        rows, skipped = [], 0
        total = len(scan_df)
        for i, srow in enumerate(scan_df.itertuples(index=False), 1):
            yield (f"{warn}⏳ ({i}/{total}) عم نحلل **{srow.sym}** ...",
                   build_table(pd.DataFrame(rows)) if rows else empty_table(), "")
            row = deep_row(srow.sym, float(srow.close), float(srow.volume))
            if row is None:
                skipped += 1
            else:
                rows.append(row)
            time.sleep(0.15)

        if not rows:
            yield f"{warn}❌ مفيش سهم اتحلل — المشكلة في جلب التاريخ", empty_table(), ""
            return

        res = pd.DataFrame(rows)
        status = (f"{warn}✅ **خلص المسح** — السوق: {n_all} سهم | فوق فلتر السيولة: {total} | "
                  f"اتحلل: {len(rows)} (اتخطى {skipped}) | ⏰ {now_cairo().strftime('%H:%M')}")

        nbuy = int((res["score"] >= 4).sum())
        nmid = int(((res["score"] >= 2) & (res["score"] < 4)).sum())
        nsell = int((res["score"] <= -2).sum())
        nneu = len(res) - nbuy - nmid - nsell

        picks = res[res["score"] >= 2].sort_values("score", ascending=False).head(3)
        if len(picks):
            plines = ["", "**🎯 أعلى الفرص:**"]
            for i, r in enumerate(picks.itertuples(), 1):
                d = r.d
                plines.append(
                    f"{i}. **{r.sym}** — {r.dec} | دخول {d['entry'][0]}–{d['entry'][1]}"
                    f" | وقف {d['stop_loss']} | هدف {d['resistance']} | R/R {d['rr']}")
        else:
            plines = ["", "**🎯 أعلى الفرص:** مفيش حاجة واضحة دلوقتي — الصبر أحسن"]

        summary = ("### 📋 الملخص\n"
                   f"🟢 شراء: **{nbuy}** | 🟡 شراء محتمل: **{nmid}** | "
                   f"⚪ محايد: **{nneu}** | 🔴 بيع: **{nsell}**\n"
                   + "\n".join(plines)
                   + "\n\n⚠️ *تحليل تعليمي — مش نصيحة مالية*")

        yield status, build_table(res), summary
    except Exception as e:
        yield f"❌ حصلت مشكلة في المسح: {e}", empty_table(), ""

# ============ 8) الرسم والتقرير ============

def make_chart(r):
    df = r["df"].tail(120)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.8, label="Price")
    ax1.plot(df.index, df["SMA50"], color="#ff7f0e", lw=1.2, label="SMA50")
    for s in r["supports"]:
        ax1.axhline(s, color="green", ls="--", lw=1, alpha=0.7)
        ax1.text(df.index[0], s, f" S{s:.1f}", color="green", fontsize=9, va="bottom")
    for res in r["resistances"]:
        ax1.axhline(res, color="red", ls="--", lw=1, alpha=0.7)
        ax1.text(df.index[0], res, f" R{res:.1f}", color="red", fontsize=9, va="bottom")
    ax1.axhspan(r["entry"][0], r["entry"][1], color="green", alpha=0.15)
    ax1.axhline(r["price"], color="#2b6cb0", ls=":", lw=1.6, label="Now")
    ax1.set_title(f"{r['symbol']} — Price / S / R")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    vals = (df["Close"] * df["Volume"]) / 1e6
    colors = ["#2ca02c" if c >= o else "#d62728"
              for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, vals, color=colors, width=1.0)
    ax2.set_ylabel("Value (M EGP)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig

def report(r):
    liq = r["liq"]
    desc = SYMBOL_DESC.get(r["symbol"], "")
    head = f"### 📊 {r['symbol']}" + (f" — *{desc}*" if desc else "")
    lines = [
        head,
        (f"**💰 السعر:** {r['price']:.2f} ج 🕒 *({r['p_time']})*"
         f" | **RSI:** {r['rsi']} | **الدرج:** {r['score']:+d}"),
        f"🗂 **المصدر:** {r['psrc']}",
    ]
    if r["stale"]:
        lines.append("⚠️ **البيانات مش من النهاردة** (إجازة أو تأخير المصدر)")
    lines += [
        f"## {r['decision']}", "---",
        f"🛡️ **الدعم:** {r['support']}",
        f"⚔️ **المقاومة (الهدف):** {r['resistance']}",
        f"🎯 **منطقة الدخول:** {r['entry'][0]} → {r['entry'][1]}",
        f"*{r['entry_note']}*",
        f"🚨 **وقف الخسارة:** {r['stop_loss']} | ⚖️ عائد/مخاطرة: {r['rr']}",
        "---",
        f"💧 **السيولة:** {liq['rate']} — متوسط {fmt_num(liq['avg_value'])} ج/يوم",
        f"📦 **الكميات:** {fmt_num(liq['avg_vol'])} سهم/يوم",
        "---", "**الأسباب:**",
    ]
    lines += [f"- {x}" for x in r["reasons"]]
    lines += ["---", "⚠️ *تحليل تعليمي — مش نصيحة مالية*"]
    return "\n".join(lines)

def app(symbol):
    try:
        r = analyze(symbol)
        return report(r), make_chart(r)
    except Exception as e:
        return f"❌ حصلت مشكلة: {e}", None

# ============ 9) منطق القايمة ============

def wl_add(new_sym, current):
    global STOCKS
    if not new_sym or not new_sym.strip():
        return current, gr.update(), "", "⚠️ اكتب رمز السهم الأول"
    s = new_sym.strip().upper()
    if s in current:
        return current, gr.update(), "", f"⚠️ {s} موجود في القايمة بالفعل"
    wl = save_watchlist(list(current) + [s])
    STOCKS = wl
    return wl, gr.update(choices=wl, value=[]), "", f"✅ اتضاف **{s}** — القايمة بقت {len(wl)} سهم"

def wl_remove(selected, current):
    global STOCKS
    if not selected:
        return current, gr.update(), "", "⚠️ علّم على الأسهم اللي عايز تحذفها الأول"
    wl = save_watchlist([s for s in current if s not in selected])
    STOCKS = wl
    return wl, gr.update(choices=wl, value=[]), "", f"🗑️ اتحذف: {', '.join(selected)} — القايمة بقت {len(wl)} سهم"

def wl_reset():
    global STOCKS
    wl = save_watchlist(DEFAULT_WATCHLIST)
    STOCKS = wl
    return wl, gr.update(choices=wl, value=[]), "", "↩️ رجعت القايمة الافتراضية"

def refresh_syms(cur):
    syms = fetch_all_symbols()
    val = cur if (cur and cur.strip().upper() in syms) else (syms[0] if syms else None)
    msg = f"✅ قايمة الأسهم محدثة — **{len(syms)} سهم** متاح"
    return gr.update(choices=syms, value=val), msg

# ============ 10) الواجهة ============

try:
    ALL_STOCKS = fetch_all_symbols()
except Exception:
    ALL_STOCKS = sorted(set(DEFAULT_WATCHLIST + FALLBACK_STOCKS))

demo = gr.Blocks(theme=gr.themes.Soft(), title="محلل البورصة المصرية")

with demo:
    gr.Markdown("## 📈 محلل البورصة المصرية — مسح السوق كامل + تحليل عميق")
    wl_state = gr.State(STOCKS)

    with gr.Tab("🕵️ مسح السوق"):
        with gr.Row():
            scope = gr.Dropdown(["كل السوق", "⭐ قايمتي"],
                                value="كل السوق", label="النطاق")
            min_liq = gr.Slider(1, 50, value=5, step=1,
                                label="أقل سيولة يومية (مليون جنيه)")
            top_n = gr.Slider(5, 25, value=15, step=1,
                                label="عدد التحليل العميق (أعلى سيولة)")
        scan_btn = gr.Button("🚀 ابدأ المسح", variant="primary")
        status = gr.Markdown()
        table = gr.Dataframe(interactive=False, wrap=True)
        summary = gr.Markdown()

    with gr.Tab("🔍 تحليل سهم واحد"):
        with gr.Row():
            sym_in = gr.Dropdown(choices=ALL_STOCKS,
                                 value="COMI" if "COMI" in ALL_STOCKS
                                 else (ALL_STOCKS[0] if ALL_STOCKS else None),
                                 label=f"اختار السهم — كل أسهم البورصة ({len(ALL_STOCKS)})",
                                 allow_custom_value=True)
            ref_btn = gr.Button("🔄 حدّث القايمة", variant="secondary")
        go_btn = gr.Button("حلّل", variant="primary")
        out_md = gr.Markdown()
        out_plot = gr.Plot()

    with gr.Tab("⭐ قايمتي"):
        gr.Markdown(
            "قايمتك المخصوصة — بتظهر كـ **نطاق مسح** في تاب 🕵️\n\n"
            "💾 القايمة بتتحفظ تلقائي طول ما التطبيق شغال")
        wl_box = gr.CheckboxGroup(choices=STOCKS, value=[],
                                  label="أسهم قايمتك — علّم على اللي عايز تحذفه")
        with gr.Row():
            wl_new = gr.Textbox(label="ضيف سهم جديد",
                                placeholder="مثال: ETEL أو CIRA",
                                max_lines=1, scale=3)
            wl_add_btn = gr.Button("➕ إضافة", scale=1)
            wl_del_btn = gr.Button("🗑️ حذف المحدد", scale=1)
            wl_res_btn = gr.Button("↩️ رجّع الافتراضي", scale=1)
        wl_status = gr.Markdown()

    scan_btn.click(run_scan, [scope, min_liq, top_n], [status, table, summary])
    go_btn.click(app, sym_in, [out_md, out_plot])
    ref_btn.click(refresh_syms, sym_in, [sym_in, out_md])
    wl_add_btn.click(wl_add, [wl_new, wl_state],
                     [wl_state, wl_box, wl_new, wl_status])
    wl_del_btn.click(wl_remove, [wl_box, wl_state],
                     [wl_state, wl_box, wl_new, wl_status])
    wl_res_btn.click(wl_reset, None,
                     [wl_state, wl_box, wl_new, wl_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
