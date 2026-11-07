"""
Aplikasi Prediksi Saham BBRI + Analisis Sentimen
Metode prediksi harga: LSTM (Long Short-Term Memory)
Sentimen komentar: lexicon-based (kata dasar baku KBBI + stemmer imbuhan ringan)
Label sentimen: hanya Positif / Negatif
"""

import io
import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Saham BBRI & Analisis Sentimen",
    page_icon="📈",
    layout="wide",
)

YF_TICKER = "BBRI.JK"

INTERVAL_PERIOD_OPTIONS = {
    "1m": ["1d", "5d"],
    "5m": ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "30m": ["1d", "5d", "1mo"],
    "60m": ["1d", "5d", "1mo"],
}

LIVE_REFRESH_SECONDS = 45

# Label cadangan: dipakai HANYA jika komentar tidak punya satu pun kata sentimen.
# Ubah ke "Positif" jika ingin sebaliknya.
LABEL_DEFAULT = "Negatif"

# ------------------------------------------------------------------
# LEXICON SENTIMEN (KATA DASAR BAKU MENURUT KBBI, KONTEKS SAHAM)
# Kata berimbuhan (menguat, melonjak, kerugian, dijual, dst.) dikenali
# lewat stemmer ringan di bawah, jadi cukup tulis kata dasarnya.
# ------------------------------------------------------------------
POSITIVE_WORDS = {
    "naik", "untung", "laba", "profit", "bullish", "kuat", "positif",
    "optimis", "optimistis", "rebound", "beli", "akumulasi", "breakout",
    "ath", "rally", "reli", "solid", "bagus", "baik", "aman", "stabil",
    "tumbuh", "hijau", "melesat", "lonjak", "roket", "apresiasi",
    "recovery", "pulih", "dividen", "growth", "mantap", "sukses",
    "kembang", "peluang", "cerah", "top up", "strong buy", "laba naik",
    "kinerja bagus", "prospek cerah", "bagus sekali",
    # --- tambahan: ucapan/ekspresi positif umum & istilah trading ---
    "goodjob", "good job", "congrats", "congratulation", "alhamdulillah",
    "hamdalah", "syukur", "gas", "gaspol", "gacor", "keren", "hebat", "jos",
    "moon", "to the moon", "uptrend", "bangkit", "borong", "serok", "senang",
    "bahagia", "lega", "puas", "berkah", "rezeki", "buy", "green", "ara",
    "undervalued", "happy", "tenang", "sehat",
    "emojipos",  # penanda internal untuk emoji positif
}

NEGATIVE_WORDS = {
    "turun", "rugi", "loss", "jeblok", "anjlok", "bearish", "lemah",
    "negatif", "pesimis", "pesimistis", "crash", "dump", "jual", "sell",
    "hindar", "risiko", "waspada", "puruk", "ambruk", "rosot", "koreksi",
    "bangkrut", "delisting", "skandal", "fraud", "tipu", "hoaks", "bohong",
    "manipulasi", "merah", "tersangkut", "krisis", "gonjang", "goyang",
    "gejolak", "khawatir", "jatuh", "drop", "lesu", "sepi", "parah",
    "buruk", "hancur", "tekan", "ancam",
    "gagal bayar", "cut loss", "panic sell", "sell off", "was was",
    # --- tambahan: ekspresi negatif umum & istilah trading ---
    "kelaut", "ke laut", "lemas", "pusing", "pening", "sedih", "nangis", "tangis",
    "ngenes", "nelangsa", "miris", "sakit", "stres", "stress", "depresi",
    "frustrasi", "kecewa", "kesal", "marah", "sial", "apes", "sialan", "bodoh",
    "goblok", "tolol", "sampah", "ampas", "terjun", "jebol", "tumbang", "rontok",
    "ambles", "longsor", "bakar", "boncos", "kandas", "remuk", "bubar", "tekor",
    "minus", "ngeri", "mengerikan", "seram", "danger", "bahaya", "panik", "cemas",
    "takut", "menyesal", "sesal", "kapok", "zonk", "gagal", "jelek", "payah",
    "bobrok", "busuk", "kacau", "bencana", "kiamat", "mampus", "burgundy",
    "maroon", "arb", "downtrend", "bleeding", "red", "dead", "trash", "fail",
    "stop loss", "rusak",
    "emojineg",  # penanda internal untuk emoji negatif
}

# Kata tidak baku -> baku (acuan KBBI). Hanya pemetaan yang maknanya
# benar-benar sama; kata ambigu (buat, oke, parah, nyangkut) TIDAK dipetakan.
SLANG_TO_BAKU = {
    "yg": "yang", "yng": "yang", "dgn": "dengan", "dg": "dengan",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "jgn": "jangan", "jg": "juga", "sm": "sama", "utk": "untuk",
    "tp": "tetapi", "tapi": "tetapi", "namun": "tetapi",
    "sayangnya": "tetapi", "trs": "terus", "trus": "terus",
    "dr": "dari", "dri": "dari", "krg": "kurang", "kurng": "kurang",
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "kaga": "tidak",
    "kagak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "enggak": "tidak", "tak": "tidak",
    "blm": "belum", "belom": "belum",
    "udah": "sudah", "udh": "sudah", "dah": "sudah", "sdh": "sudah",
    "skrg": "sekarang", "skrng": "sekarang", "sekrang": "sekarang",
    "bgt": "sangat", "bngt": "sangat", "banget": "sangat",
    "byk": "banyak", "bnyk": "banyak", "hrs": "harus",
    "bs": "bisa", "bsa": "bisa",
    "mantul": "mantap", "mantab": "mantap", "mantep": "mantap",
    "kece": "bagus", "cakep": "bagus", "bgus": "bagus", "bagu": "bagus",
    "cuan": "untung", "cuannn": "untung",
    "ambrol": "ambruk", "amblas": "ambruk", "ancur": "hancur",
    "nyangkut": "tersangkut", "tersangkutt": "tersangkut",
    "gaskeun": "beli", "moga": "semoga", "smoga": "semoga",
    "resiko": "risiko", "hoax": "hoaks", "optimistis": "optimis",
    "pesimistis": "pesimis",
    # tambahan
    "ijo": "hijau", "gj": "goodjob", "gudjob": "goodjob",
    "alhamdulilah": "alhamdulillah", "alhamdullilah": "alhamdulillah",
    "alhamdulillaah": "alhamdulillah",
    "bgs": "bagus", "kzl": "kesal", "burgundi": "burgundy",
}

NEGATION_WORDS = {"tidak", "bukan", "jangan", "belum", "kurang"}
CONTRAST_WORDS = {"tetapi", "melainkan"}
INTENSIFIERS = {"sangat", "amat", "sekali", "terlalu"}


# Emoji yang jelas bermakna positif / negatif. Emoji ambigu (😂 🤣 🔥 🙏 dll.)
# sengaja TIDAK dimasukkan karena maknanya bergantung konteks.
# (simbol BMP ditulis dengan kode \u agar tidak terbawa variation selector)
_POS_EMOJI = set("😇😍🥰😊😁😀😃😄🤩😎🎉🥳👏👍💪🚀🌕🤑💰💚🟢📈✅💕😘😻") | {"\u2764", "\u2665"}
_NEG_EMOJI = set("😭😢😞😔😩😫😡🤬😱😨😰😖😣😥😓💀👎📉🔴🩸🤡🤮🤢💔😤😵🔻🪦💸") | {"\u2620", "\u26B0"}
_EMOJI_LABEL = {"emojipos": "emoji positif", "emojineg": "emoji negatif"}


def _emoji_to_tokens(text: str) -> str:
    """Ganti emoji bermakna jelas menjadi token 'emojipos' / 'emojineg' agar ikut dihitung."""
    out = []
    for ch in text:
        if ch in _POS_EMOJI:
            out.append(" emojipos ")
        elif ch in _NEG_EMOJI:
            out.append(" emojineg ")
        else:
            out.append(ch)
    return "".join(out)


def clean_text(text: str) -> str:
    text = _emoji_to_tokens(str(text)).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # "mantapppp" -> "mantap"
    words = [re.sub(r"(.)\1{2,}", r"\1", w) for w in text.split()]
    hasil = []
    for w in words:
        if w in SLANG_TO_BAKU:
            hasil.append(SLANG_TO_BAKU[w])
        # huruf akhir dobel ("ijoo" -> "ijo")
        elif len(w) > 3 and w[-1] == w[-2] and w[:-1] in SLANG_TO_BAKU:
            hasil.append(SLANG_TO_BAKU[w[:-1]])
        else:
            hasil.append(w)
    return " ".join(hasil)


_PREFIXES = ("memper", "diper", "meng", "meny", "men", "mem", "me",
             "peng", "peny", "pen", "pem", "pe", "ber", "ter", "per",
             "di", "ke", "se")
_SUFFIXES = ("kan", "nya", "lah", "kah", "an", "i")


def _candidates(tok: str) -> list:
    """Bentuk kata dasar yang mungkin dari sebuah kata berimbuhan (stemmer ringan).
    Hasilnya hanya dipakai untuk dicocokkan dengan kamus, jadi aman dari salah-stem."""
    stems = [tok]
    if len(tok) > 3 and tok[-1] == tok[-2]:  # "digaskann" -> "digaskan"
        stems.append(tok[:-1])
    for base in list(stems):
        for suf in _SUFFIXES:
            if base.endswith(suf) and len(base) - len(suf) >= 3:
                stems.append(base[: -len(suf)])
    cands = list(stems)
    for s in stems:
        for pre in _PREFIXES:
            if s.startswith(pre) and len(s) - len(pre) >= 3:
                r = s[len(pre):]
                cands.append(r)
                if pre in ("meny", "peny"):
                    cands.append("s" + r)
                if pre in ("meng", "peng"):
                    cands.append("k" + r)
                if pre in ("mem", "pem"):
                    cands.append("p" + r)
                if pre in ("men", "pen"):
                    cands.append("t" + r)
    return cands


_POS_SINGLE = {w for w in POSITIVE_WORDS if " " not in w}
_NEG_SINGLE = {w for w in NEGATIVE_WORDS if " " not in w}
_POS_PHRASES = [w.split() for w in POSITIVE_WORDS if " " in w]
_NEG_PHRASES = [w.split() for w in NEGATIVE_WORDS if " " in w]


def _analyze(tokens, negation_window=2):
    """Hitung bobot positif/negatif satu segmen + daftar kata penentu + posisi tiap kata."""
    pos, neg, matched, hits = 0, 0, [], []
    used = set()

    def negated(i):
        return any(tokens[j] in NEGATION_WORDS for j in range(max(0, i - negation_window), i))

    def weight(i, n=1):
        before = tokens[i - 1] if i > 0 else ""
        after = tokens[i + n] if i + n < len(tokens) else ""
        return 2 if (before in INTENSIFIERS or after in INTENSIFIERS) else 1

    def add(is_pos, i, n, label):
        nonlocal pos, neg
        w = weight(i, n)
        flip = negated(i)
        final_pos = is_pos != flip
        if final_pos:
            pos += w
        else:
            neg += w
        shown = _EMOJI_LABEL.get(label, label)
        matched.append(("+" if final_pos else "-") + (f"tidak {shown}" if flip else shown))
        hits.append((i, final_pos))
        used.update(range(i, i + n))

    # 1) frasa dulu (agar kata di dalam frasa tidak dihitung dua kali)
    for phrases, is_pos in ((_POS_PHRASES, True), (_NEG_PHRASES, False)):
        for ph in phrases:
            n = len(ph)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == ph and not used.intersection(range(i, i + n)):
                    add(is_pos, i, n, " ".join(ph))

    # 2) kata tunggal (dengan stemming imbuhan)
    for i, tok in enumerate(tokens):
        if i in used:
            continue
        cands = _candidates(tok)
        in_pos = any(c in _POS_SINGLE for c in cands)
        in_neg = any(c in _NEG_SINGLE for c in cands)
        if in_pos == in_neg:  # tidak cocok / ambigu -> abaikan
            continue
        add(in_pos, i, 1, tok)
    return pos, neg, matched, hits


def score_sentiment(text: str):
    """Return (skor, label, kata_penentu, keyakinan). Label hanya Positif / Negatif."""
    tokens = clean_text(text).split()

    # Kalimat kontras ("bagus tetapi berisiko"): bagian setelah 'tetapi' lebih menentukan.
    segments, cur = [], []
    for t in tokens:
        if t in CONTRAST_WORDS:
            segments.append(cur)
            cur = []
        else:
            cur.append(t)
    segments.append(cur)

    results = [_analyze(seg) for seg in segments]
    last_pos, last_neg = results[-1][0], results[-1][1]
    if len(results) > 1 and (last_pos or last_neg):
        pos, neg = last_pos, last_neg
        matched = results[-1][2]
    else:
        pos = sum(r[0] for r in results)
        neg = sum(r[1] for r in results)
        matched = [m for r in results for m in r[2]]

    score = pos - neg
    if score > 0:
        return score, "Positif", ", ".join(matched), "Jelas"
    if score < 0:
        return score, "Negatif", ", ".join(matched), "Jelas"

    # Skor 0 -> tentukan dari kata sentimen paling akhir (kesimpulan penulis)
    for r in reversed(results):
        if r[3]:
            _, final_pos = max(r[3], key=lambda h: h[0])
            label = "Positif" if final_pos else "Negatif"
            return score, label, ", ".join(matched), "Sedang (seri, ikut kata terakhir)"

    # Tidak ada kata sentimen sama sekali
    return score, LABEL_DEFAULT, "-", "Lemah (tanpa kata sentimen)"


# ------------------------------------------------------------------
# LOAD DATA (HISTORIS, DARI FILE YANG DIUPLOAD USER)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ohlcv(file):
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Movement"] = df["Close"].diff().apply(
        lambda x: "Naik" if x > 0 else ("Turun" if x < 0 else "Stabil")
    )
    return df


@st.cache_data(show_spinner=False)
def load_comments(file):
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    df["PostDate"] = pd.to_datetime(df["PostDate"])
    hasil = df["TweetText"].apply(score_sentiment)
    df["SentimentScore"] = [h[0] for h in hasil]
    df["Sentiment"] = [h[1] for h in hasil]
    df["KataPenentu"] = [h[2] for h in hasil]
    df["Keyakinan"] = [h[3] for h in hasil]
    # --- untuk EDA panjang komentar ---
    teks = df["TweetText"].astype(str)
    df["JumlahKarakter"] = teks.str.len()
    df["JumlahKata"] = teks.str.split().str.len()
    return df.sort_values("PostDate").reset_index(drop=True)


# ------------------------------------------------------------------
# DATA HARGA REAL-TIME (YAHOO FINANCE via yfinance)
# ------------------------------------------------------------------
@st.cache_data(ttl=LIVE_REFRESH_SECONDS, show_spinner=False)
def fetch_live_bbri(period: str = "1d", interval: str = "1m"):
    max_retries = 2
    last_err = None
    for attempt in range(max_retries):
        try:
            hist = yf.Ticker(YF_TICKER).history(period=period, interval=interval)
            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                date_col = "Datetime" if "Datetime" in hist.columns else "Date"
                hist = hist.rename(columns={date_col: "Date"})
                return hist[["Date", "Open", "High", "Low", "Close", "Volume"]], None
            last_err = "Yahoo Finance mengembalikan data kosong untuk kombinasi periode/interval ini."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries - 1:
            time.sleep(1.5)
    return None, last_err


# ------------------------------------------------------------------
# LSTM MODEL
# ------------------------------------------------------------------
def create_sequences(scaled_data, window):
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i - window:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)


def validate_lstm_params(n_rows, window, train_ratio):
    train_size = int(n_rows * train_ratio)
    test_size = n_rows - train_size

    if n_rows < window + 20:
        return (
            f"Data harga terlalu sedikit ({n_rows} baris) untuk window={window} hari. "
            f"Tambahkan lebih banyak data historis, atau kecilkan nilai Window di sidebar."
        )
    if train_size <= window:
        return (
            f"Proporsi Data Latih terlalu kecil dibanding Window (window={window} hari, "
            f"data latih hanya {train_size} baris). Perbesar Proporsi Data Latih atau "
            f"perkecil Window."
        )
    if test_size <= window:
        return (
            f"Proporsi Data Latih terlalu besar sehingga data uji tidak cukup "
            f"(hanya {test_size} baris, butuh lebih dari {window}). Kecilkan Proporsi Data Latih."
        )
    return None


@st.cache_resource(show_spinner=False)
def train_lstm(close_values, window, train_ratio, epochs, batch_size, units, dropout):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.random.set_seed(42)

    values = close_values.reshape(-1, 1)
    train_size = int(len(values) * train_ratio)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:train_size])
    scaled = scaler.transform(values)

    train_data = scaled[:train_size]
    test_data = scaled[train_size - window:]

    X_train, y_train = create_sequences(train_data, window)
    X_test, y_test = create_sequences(test_data, window)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(window, 1)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0))
    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return {
        "model": model,
        "scaler": scaler,
        "scaled_full": scaled,
        "train_size": train_size,
        "window": window,
        "train_pred": train_pred.flatten(),
        "test_pred": test_pred.flatten(),
        "y_train_actual": y_train_actual.flatten(),
        "y_test_actual": y_test_actual.flatten(),
    }


def forecast_future(result, n_days):
    model = result["model"]
    scaler = result["scaler"]
    scaled_full = result["scaled_full"]
    window = result["window"]

    last_window = scaled_full[-window:].reshape(1, window, 1)
    preds_scaled = []
    current = last_window.copy()
    for _ in range(n_days):
        next_val = model.predict(current, verbose=0)[0, 0]
        preds_scaled.append(next_val)
        current = np.append(current[:, 1:, :], [[[next_val]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


def build_forecast_figure(price_df, future_dates, future_preds, n_hist=60, height=450, title=None,
                          back_dates=None, back_preds=None):
    """Grafik historis + prediksi yang SAMBUNG (garis prediksi mulai dari titik terakhir historis)."""
    hist = price_df.tail(n_hist)
    last_date = pd.to_datetime(hist["Date"].iloc[-1])
    last_price = float(hist["Close"].iloc[-1])

    conn_x = [last_date] + list(pd.to_datetime(future_dates))
    conn_y = [last_price] + [float(v) for v in future_preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"],
        name=f"Harga Historis ({n_hist} hari terakhir)",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=conn_x, y=conn_y,
        name="Prediksi ke Depan",
        mode="lines+markers",
        line=dict(color="#ff7f0e", width=2),
        marker=dict(size=[0] + [8] * len(future_preds)),
        hovertemplate="%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra>Prediksi</extra>",
    ))
    if back_dates is not None and back_preds is not None and len(back_preds):
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(back_dates), y=[float(v) for v in back_preds],
            name=f"Prediksi Model ({len(back_preds)} Hari Sebelumnya)",
            mode="lines+markers",
            line=dict(color="#9467bd", width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra>Prediksi (hari lalu)</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[last_date], y=[last_price],
        mode="markers", name="Harga Aktual Terakhir",
        marker=dict(color="#d62728", size=10, line=dict(color="white", width=1.5)),
        hovertemplate="%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra>Aktual</extra>",
    ))

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    y_all = [float(v) for v in hist["Close"]] + conn_y
    if back_preds is not None and len(back_preds):
        y_all += [float(v) for v in back_preds]
    pad = (max(y_all) - min(y_all)) * 0.08 or 1
    fig.update_yaxes(range=[min(y_all) - pad, max(y_all) + pad])

    fig.update_layout(
        title=title, xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
        height=height, hovermode="x unified",
    )
    return fig


def add_close_price(comments, prices_df):
    """Tambah kolom 'Harga Saham (Close)' pada tanggal komentar, lengkap dengan
    tanda naik/turun di sampingnya (dibanding Close hari bursa sebelumnya).
    Jika tanggal komentar bukan hari bursa (akhir pekan/libur), dipakai harga
    hari bursa terakhir sebelum tanggal tersebut. Jika Close sama dengan hari
    sebelumnya, ditandai 'Stabil'."""
    prices = prices_df[["Date", "Close", "Movement"]].copy()
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize().astype("datetime64[ns]")
    prices = prices.rename(columns={"Date": "_tgl", "Close": "_close", "Movement": "_arah"})
    prices = prices.sort_values("_tgl")

    c = comments.copy()
    c["_tgl"] = pd.to_datetime(c["PostDate"]).dt.normalize().astype("datetime64[ns]")
    c = c.sort_values("_tgl")

    merged = pd.merge_asof(c, prices, on="_tgl", direction="backward")

    tanda = {"Naik": " 📈 Naik", "Turun": " 📉 Turun", "Stabil": " ➖ Stabil"}

    def _fmt(row):
        if pd.isna(row["_close"]):
            return ""
        harga = f"Rp{row['_close']:,.0f}".replace(",", ".")
        return harga + tanda.get(row["_arah"], "")

    merged["Harga Saham (Close)"] = merged.apply(_fmt, axis=1)
    merged = merged.drop(columns=["_tgl", "_close", "_arah"])
    return merged.sort_values("PostDate", ascending=False)


def show_table(df, height=None, **kwargs):
    df_show = df.reset_index(drop=True).copy()
    df_show.index = df_show.index + 1
    df_show.index.name = "No"
    if height is not None:
        st.dataframe(df_show, use_container_width=True, height=height, **kwargs)
    else:
        st.dataframe(df_show, use_container_width=True, **kwargs)


def build_komentar_excel(df, sentimen_sejalan):
    """Buat file Excel (.xlsx) berisi daftar komentar + ringkasan jumlah Positif,
    Negatif, dan Total yang dihitung dengan RUMUS Excel (COUNTIF), sehingga ikut
    berubah otomatis kalau isi kolom Sentiment diedit di Excel."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    d = df.reset_index(drop=True).copy()
    # Hilangkan emoji agar rapi di Excel
    if "Harga Saham (Close)" in d.columns:
        d["Harga Saham (Close)"] = (
            d["Harga Saham (Close)"].astype(str)
            .str.replace("📈 ", "", regex=False)
            .str.replace("📉 ", "", regex=False)
            .str.replace("➖ ", "", regex=False)
        )

    wb = Workbook()
    ws = wb.active
    ws.title = "Komentar"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ---- Tabel komentar (kolom A-E) ----
    headers = ["No"] + list(d.columns)
    for c_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c_idx, value=h)
        cell.fill, cell.font, cell.border = header_fill, header_font, border
        cell.alignment = Alignment(horizontal="center", vertical="center")

    sent_col_idx = headers.index("Sentiment") + 1
    sent_col = get_column_letter(sent_col_idx)

    for r_idx, row in enumerate(d.itertuples(index=False), start=2):
        ws.cell(row=r_idx, column=1, value=r_idx - 1).border = border
        for c_idx, val in enumerate(row, start=2):
            if isinstance(val, pd.Timestamp):
                val = val.to_pydatetime()
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.border = border
            if headers[c_idx - 1] == "PostDate":
                cell.number_format = "yyyy-mm-dd"
            if headers[c_idx - 1] == "TweetText":
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    last_row = max(len(d) + 1, 2)
    widths = {"No": 6, "Username": 22, "PostDate": 13, "Harga Saham (Close)": 20,
              "TweetText": 70, "Sentiment": 12}
    for c_idx, h in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(c_idx)].width = widths.get(h, 15)
    ws.freeze_panes = "A2"

    # ---- Ringkasan (di kanan tabel), pakai rumus Excel ----
    rng = f"${sent_col}$2:${sent_col}${last_row}"
    label_col = len(headers) + 2      # sisakan 1 kolom kosong sebagai pemisah
    value_col = label_col + 1
    vl = get_column_letter(value_col)
    ws.cell(row=1, column=label_col, value="Ringkasan Sentimen")
    ws.cell(row=1, column=value_col, value="Nilai")
    for c in (label_col, value_col):
        cell = ws.cell(row=1, column=c)
        cell.fill, cell.font, cell.border = header_fill, header_font, border
        cell.alignment = Alignment(horizontal="center")

    summary_rows = [
        ("Komentar Positif", f'=COUNTIF({rng},"Positif")', None),
        ("Komentar Negatif", f'=COUNTIF({rng},"Negatif")', None),
        ("Total (Positif + Negatif)", f"={vl}2+{vl}3", None),
        ("Sentimen sejalan dengan prediksi", sentimen_sejalan, None),
        ("Jumlah komentar sejalan", f'=IF({vl}5="Positif",{vl}2,{vl}3)', None),
        ("Persentase sejalan", f"=IF({vl}4=0,0,{vl}6/{vl}4)", "0.0%"),
    ]
    for i, (label, val, fmt) in enumerate(summary_rows, start=2):
        lc = ws.cell(row=i, column=label_col, value=label)
        vc = ws.cell(row=i, column=value_col, value=val)
        lc.border = vc.border = border
        vc.alignment = Alignment(horizontal="center")
        if fmt:
            vc.number_format = fmt
    ws.cell(row=4, column=label_col).font = Font(bold=True)
    ws.cell(row=4, column=value_col).font = Font(bold=True)
    ws.column_dimensions[get_column_letter(label_col - 1)].width = 3
    ws.column_dimensions[get_column_letter(label_col)].width = 34
    ws.column_dimensions[vl].width = 14

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def mape(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.mean(np.abs((a - b) / a)) * 100)


# ------------------------------------------------------------------
# PREPROCESSING TIME SERIES (MISSING VALUE & OUTLIER)
# ------------------------------------------------------------------
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def check_missing_values(df):
    """Hitung NaN dan nilai 0 (tidak valid untuk harga) per kolom."""
    rows = []
    for c in [c for c in PRICE_COLS if c in df.columns]:
        n_nan = int(df[c].isna().sum())
        n_zero = int((df[c] == 0).sum()) if c != "Volume" else 0
        total = n_nan + n_zero
        rows.append({
            "Kolom": c,
            "Missing (NaN)": n_nan,
            "Nilai 0 (tidak valid)": n_zero,
            "Total Bermasalah": total,
            "Persentase (%)": total / len(df) * 100 if len(df) else 0.0,
        })
    return pd.DataFrame(rows)


def check_time_gaps(df):
    """Duplikat tanggal & hari kerja (Sen-Jum) yang tidak ada datanya."""
    dates = pd.to_datetime(df["Date"]).dt.normalize()
    n_dup = int(dates.duplicated().sum())
    expected = pd.bdate_range(dates.min(), dates.max())
    n_gap = len(expected.difference(pd.DatetimeIndex(dates.unique())))
    return n_dup, n_gap


def handle_missing_values(df):
    """Hapus tanggal kosong/duplikat, ubah harga <= 0 jadi NaN,
    lalu isi NaN dengan interpolasi berbasis waktu (+ ffill/bfill di ujung)."""
    d = df.dropna(subset=["Date"]).copy()
    d = d.drop_duplicates(subset="Date", keep="last").sort_values("Date")
    cols = [c for c in PRICE_COLS if c in d.columns]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        if c != "Volume":
            d.loc[d[c] <= 0, c] = np.nan
    d = d.set_index("Date")
    d[cols] = d[cols].interpolate(method="time", limit_direction="both")
    d[cols] = d[cols].ffill().bfill()
    return d.reset_index()


def detect_outliers_iqr(df, k=1.5):
    """Outlier dideteksi dari return harian Close dengan aturan IQR
    (lebih tepat untuk time series daripada IQR harga mentah yang ikut tren)."""
    ret = df["Close"].pct_change()
    q1, q3 = ret.quantile(0.25), ret.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = ((ret < lower) | (ret > upper)).fillna(False)
    return mask, ret, lower, upper


def handle_outliers(df, mask, method):
    d = df.copy()
    if method == "Tidak ditangani" or not mask.any():
        return d
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in d.columns]
    if method == "Interpolasi linear":
        d.loc[mask, cols] = np.nan
        d[cols] = d[cols].interpolate(method="linear", limit_direction="both")
    else:  # Median bergulir (window 5)
        med = d[cols].rolling(5, center=True, min_periods=1).median()
        d.loc[mask, cols] = med.loc[mask, cols]
    return d


# ------------------------------------------------------------------
# SIDEBAR - NAVIGASI & DATA
# ------------------------------------------------------------------
st.sidebar.title("📊 Menu")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Beranda",
        "📈 Prediksi Harga (LSTM)",
        "💬 Analisis Sentimen Komentar",
        "📌 Kesimpulan Akhir",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Sumber Data")
ohlcv_file = st.sidebar.file_uploader("Upload data harga (OHLCV) — wajib (.xlsx)", type=["xlsx"])
comments_file = st.sidebar.file_uploader("Upload data komentar — wajib (.xlsx)", type=["xlsx"])

if ohlcv_file is None or comments_file is None:
    st.title("📈 Prediksi Saham BBRI & Analisis Sentimen")
    st.warning(
        "Silakan upload **kedua** file data terlebih dahulu di sidebar sebelah kiri:\n\n"
        "1. Data harga (OHLCV) — file `.xlsx` dengan kolom minimal `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.\n"
        "2. Data komentar — file `.xlsx` dengan kolom minimal `Username`, `PostDate`, `TweetText`.\n\n"
        "Aplikasi tidak lagi menggunakan dataset bawaan/default — semua data diambil dari file yang diupload."
    )
    st.stop()

price_df = load_ohlcv(ohlcv_file)
comments_df = load_comments(comments_file)

current_data_signature = (
    getattr(ohlcv_file, "name", None),
    getattr(ohlcv_file, "size", None),
    len(price_df),
    float(price_df["Close"].iloc[-1]) if len(price_df) else None,
)
if st.session_state.get("lstm_data_signature") != current_data_signature:
    st.session_state.pop("lstm_result", None)
    st.session_state.pop("lstm_params", None)
    st.session_state["lstm_data_signature"] = current_data_signature

st.sidebar.caption(
    f"Data harga: {len(price_df)} baris ({price_df['Date'].min().date()} – {price_df['Date'].max().date()})\n\n"
    f"Data komentar: {len(comments_df)} baris ({comments_df['PostDate'].min().date()} – {comments_df['PostDate'].max().date()})"
)

# Kolom yang ditampilkan pada tabel komentar berlabel
KOLOM_SENTIMEN = ["Username", "PostDate", "TweetText", "Sentiment"]

# ------------------------------------------------------------------
# HALAMAN: BERANDA
# ------------------------------------------------------------------
if page == "🏠 Beranda":
    st.title("📈 Prediksi Saham BBRI & Analisis Sentimen Komentar")
    st.markdown(
        """
        Aplikasi ini menggabungkan dua hal:

        1. **Prediksi harga saham BBRI** menggunakan model **LSTM (Long Short-Term Memory)**
           berdasarkan data historis harga *Close*.
        2. **Analisis sentimen komentar** (positif / negatif) dari kumpulan komentar
           terkait saham & trading, menggunakan pendekatan **kamus kata baku KBBI (lexicon-based)**.
        """
    )

    _live_snapshot, _live_err = fetch_live_bbri()
    if _live_snapshot is not None:
        st.session_state["last_good_live_df"] = _live_snapshot
        st.session_state["last_good_live_time"] = pd.Timestamp.now()

    col1, col2, col3, col4 = st.columns(4)
    if _live_snapshot is not None and len(_live_snapshot) >= 2:
        last_close = _live_snapshot["Close"].iloc[-1]
        prev_close = _live_snapshot["Close"].iloc[-2]
        delta = last_close - prev_close
        col1.metric("Harga Terakhir (Live)", f"Rp{last_close:,.0f}", f"{delta:+.0f}")
    else:
        last_close = price_df["Close"].iloc[-1]
        prev_close = price_df["Close"].iloc[-2]
        delta = last_close - prev_close
        col1.metric("Harga Terakhir (Close)", f"Rp{last_close:,.0f}", f"{delta:+.0f}")
    col2.metric("Jumlah Data Harga", f"{len(price_df)} hari")
    col3.metric("Jumlah Komentar", f"{len(comments_df)}")
    pos_pct = (comments_df["Sentiment"] == "Positif").mean() * 100
    col4.metric("Komentar Positif", f"{pos_pct:.1f}%")

    st.subheader("Riwayat Harga Close BBRI (Real-time — Yahoo Finance)")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    with ctrl_col1:
        live_interval = st.selectbox(
            "Interval", list(INTERVAL_PERIOD_OPTIONS.keys()), index=0, key="live_interval"
        )
    with ctrl_col2:
        valid_periods = INTERVAL_PERIOD_OPTIONS.get(live_interval, ["1d"])
        live_period = st.selectbox(
            "Periode", valid_periods, index=0, key="live_period"
        )
    with ctrl_col3:
        st.write("")
        if st.button("🔄 Refresh Sekarang"):
            fetch_live_bbri.clear()
            st.rerun()

    @st.fragment(run_every=LIVE_REFRESH_SECONDS)
    def render_live_price_chart():
        interval = st.session_state.get("live_interval", "1m")
        period = st.session_state.get("live_period", "1d")
        live_df, err = fetch_live_bbri(period=period, interval=interval)

        if live_df is not None and not live_df.empty:
            st.session_state["last_good_live_df"] = live_df
            st.session_state["last_good_live_time"] = pd.Timestamp.now()
        else:
            last_good = st.session_state.get("last_good_live_df")
            if last_good is not None:
                live_df = last_good
                last_good_time = st.session_state.get("last_good_live_time")
                ts_str = last_good_time.strftime("%H:%M:%S") if last_good_time is not None else "-"
                st.warning(
                    f"Gagal mengambil data terbaru dari Yahoo Finance "
                    f"({err or 'penyebab tidak diketahui'}). Menampilkan data live "
                    f"terakhir yang berhasil diambil pada {ts_str}. Ini sering terjadi "
                    f"karena rate limit Yahoo Finance saat request terlalu sering — "
                    f"coba lagi dalam ~1 menit."
                )
            else:
                st.warning(
                    f"Gagal mengambil data real-time dari Yahoo Finance "
                    f"({err or 'cek koneksi internet atau coba periode/interval lain'}). "
                    f"Menampilkan data historis sebagai fallback."
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df["Date"], y=price_df["Close"],
                    mode="lines", name="Harga Close (Historis)", line=dict(color="#1f77b4"),
                ))
                fig.update_layout(xaxis_title="Tanggal", yaxis_title="Harga (Rp)", height=450)
                st.plotly_chart(fig, use_container_width=True)
                return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=live_df["Date"], y=live_df["Close"],
            mode="lines", name="Harga Close (Live)", line=dict(color="#1f77b4"),
        ))
        fig.update_layout(xaxis_title="Waktu", yaxis_title="Harga (Rp)", height=450)
        st.plotly_chart(fig, use_container_width=True)

        last_update = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        last_price = live_df["Close"].iloc[-1]
        st.caption(
            f"Sumber: Yahoo Finance ({YF_TICKER}) · Harga terakhir: Rp{last_price:,.0f} · "
            f"Update terakhir: {last_update} · Auto-refresh tiap {LIVE_REFRESH_SECONDS} detik."
        )

    render_live_price_chart()

    # ----------------------------------------------------------------
    # SEKILAS DATA
    # ----------------------------------------------------------------
    st.subheader("Sekilas Data")

    sekilas_min_date = min(price_df["Date"].min().date(), comments_df["PostDate"].min().date())
    sekilas_max_date = max(price_df["Date"].max().date(), comments_df["PostDate"].max().date())

    col_dari, col_sampai = st.columns(2)
    with col_dari:
        sekilas_start = st.date_input(
            "🗓️ Dari tanggal",
            value=st.session_state.get("sekilas_start_value", sekilas_max_date),
            min_value=sekilas_min_date,
            max_value=sekilas_max_date,
            key="sekilas_start",
        )
    with col_sampai:
        sekilas_end = st.date_input(
            "🗓️ Sampai tanggal",
            value=st.session_state.get("sekilas_end_value", sekilas_max_date),
            min_value=sekilas_min_date,
            max_value=sekilas_max_date,
            key="sekilas_end",
        )

    st.session_state["sekilas_start_value"] = sekilas_start
    st.session_state["sekilas_end_value"] = sekilas_end

    if sekilas_start > sekilas_end:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
        sekilas_start, sekilas_end = sekilas_end, sekilas_start

    price_df_sekilas = price_df[
        (price_df["Date"].dt.date >= sekilas_start) & (price_df["Date"].dt.date <= sekilas_end)
    ].sort_values("Date", ascending=False)

    comments_df_sekilas = comments_df[
        (comments_df["PostDate"].dt.date >= sekilas_start) & (comments_df["PostDate"].dt.date <= sekilas_end)
    ].sort_values("PostDate", ascending=False)

    label_tanggal = (
        f"**{sekilas_start}**" if sekilas_start == sekilas_end
        else f"**{sekilas_start}** – **{sekilas_end}**"
    )

    tab1, tab2 = st.tabs(["Data Harga", "Data Komentar"])
    with tab1:
        if len(price_df_sekilas):
            show_table(price_df_sekilas, height=400)
        else:
            st.info(f"Tidak ada data harga pada tanggal {label_tanggal} (mungkin hari libur/akhir pekan bursa).")
        st.caption(f"Menampilkan seluruh **{len(price_df_sekilas)}** baris data harga pada {label_tanggal}.")
    with tab2:
        if len(comments_df_sekilas):
            show_table(comments_df_sekilas[["Username", "PostDate", "TweetText"]], height=400)
        else:
            st.info(f"Tidak ada komentar pada tanggal {label_tanggal}.")
        st.caption(
            f"Menampilkan seluruh **{len(comments_df_sekilas)}** dari {len(comments_df)} "
            f"komentar pada {label_tanggal}."
        )

# ------------------------------------------------------------------
# HALAMAN: PREDIKSI HARGA (LSTM)
# ------------------------------------------------------------------
elif page == "📈 Prediksi Harga (LSTM)":
    st.title("📈 Prediksi Harga Saham BBRI dengan LSTM")

    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Parameter Model LSTM")
        window = st.slider("Window (lookback, hari)", 2, 120, 60, step=1)
        train_ratio = st.slider("Proporsi Data Latih", 0.5, 0.95, 0.8, step=0.05)
        units = st.select_slider("Jumlah Unit LSTM", options=[10, 16, 32, 50, 64, 100], value=50)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        # Epochs: kelipatan 5 sampai 100 tetap ada, ditambah pilihan 200
        epochs = st.select_slider(
            "Epochs",
            options=list(range(5, 101, 5)) + [200],
            value=25,
        )
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
        forecast_days = st.slider("Prediksi ke depan (hari)", 1, 30, 7)
        train_btn = st.button("🚀 Latih Model & Prediksi", type="primary")

    if not train_btn and "lstm_result" not in st.session_state:
        st.info(
            "Atur parameter di sidebar lalu klik **🚀 Latih Model & Prediksi** untuk memulai.\n\n"
            "Catatan: setiap kali dataset harga (OHLCV) diganti/diupload ulang, model perlu "
            "dilatih ulang dengan menekan tombol ini."
        )
        st.stop()

    if train_btn:
        validation_error = validate_lstm_params(len(price_df), window, train_ratio)
        if validation_error:
            st.error(f"❌ {validation_error}")
            st.stop()
        with st.spinner("Melatih model LSTM... ini bisa memakan waktu beberapa saat."):
            close_values = price_df["Close"].values.astype(float)
            result = train_lstm(close_values, window, train_ratio, epochs, batch_size, units, dropout)
            st.session_state["lstm_result"] = result
            st.session_state["lstm_params"] = dict(window=window, forecast_days=forecast_days)

    result = st.session_state["lstm_result"]
    window = st.session_state["lstm_params"]["window"]
    # Jumlah hari prediksi mengikuti slider secara langsung (model tidak perlu dilatih ulang,
    # karena prediksi ke depan hanya memakai model yang sudah ada).
    st.session_state["lstm_params"]["forecast_days"] = forecast_days

    train_size = result["train_size"]
    dates = price_df["Date"].values
    train_dates = dates[window:train_size]
    test_dates = dates[train_size:]

    rmse_test = rmse(result["y_test_actual"], result["test_pred"])
    mae_test = mean_absolute_error(result["y_test_actual"], result["test_pred"])
    mape_test = mape(result["y_test_actual"], result["test_pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (Data Uji)", f"{rmse_test:,.2f}")
    c2.metric("MAE (Data Uji)", f"{mae_test:,.2f}")
    c3.metric("MAPE (Data Uji)", f"{mape_test:.2f}%")

    # ---- Grafik Aktual vs Prediksi: hanya 2 warna (hijau = aktual, merah = prediksi) ----
    fig = go.Figure()
    # Harga aktual -> HIJAU
    fig.add_trace(go.Scatter(
        x=dates, y=price_df["Close"], name="Harga Aktual",
        line=dict(color="#2ca02c", width=2),
    ))
    # Prediksi (gabungan train + test) -> MERAH, satu garis, satu legenda
    pred_dates_all = np.concatenate([train_dates, test_dates])
    pred_values_all = np.concatenate([result["train_pred"], result["test_pred"]])
    fig.add_trace(go.Scatter(
        x=pred_dates_all, y=pred_values_all, name="Prediksi",
        line=dict(color="#d62728", width=2),
    ))
    fig.update_layout(title="Aktual vs Prediksi Harga Close", xaxis_title="Tanggal",
                      yaxis_title="Harga (Rp)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🔮 Prediksi {forecast_days} Hari ke Depan")
    with st.spinner("Menghitung prediksi ke depan..."):
        future_preds = forecast_future(result, forecast_days)
    last_date = pd.to_datetime(price_df["Date"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

    prev_prices = np.concatenate([[float(price_df["Close"].iloc[-1])], future_preds[:-1]])
    pred_round = np.round(future_preds)
    prev_round = np.round(prev_prices)
    change_rp = pred_round - prev_round
    arah_harga = np.where(
        change_rp > 0, "Naik 📈",
        np.where(change_rp < 0, "Turun 📉", "Stabil ➖"),
    )

    future_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi Close": future_preds,
        "Arah": arah_harga,
    })
    # Jumlah hari sebelumnya = jumlah hari prediksi ke depan (selaras dengan slider)
    n_back = min(forecast_days, len(result["test_pred"]))
    back_dates = price_df["Date"].iloc[-n_back:].reset_index(drop=True)
    back_pred = result["test_pred"][-n_back:]
    back_actual = result["y_test_actual"][-n_back:]

    fig2 = build_forecast_figure(
        price_df, future_df["Tanggal"], future_df["Prediksi Close"],
        n_hist=60, height=450, title="Proyeksi Harga ke Depan",
        back_dates=back_dates, back_preds=back_pred,
    )
    st.plotly_chart(fig2, use_container_width=True)
    future_display = future_df.copy()
    future_display["Tanggal"] = future_display["Tanggal"].dt.strftime("%Y-%m-%d")
    future_display.index = range(1, len(future_display) + 1)
    future_display.index.name = "No"
    st.dataframe(
        future_display.style.format({"Prediksi Close": "Rp{:,.0f}"}),
        use_container_width=True,
    )

    st.subheader(f"🕒 Prediksi Model {n_back} Hari Sebelumnya vs Harga Aktual")
    back_df = pd.DataFrame({
        "Tanggal": pd.to_datetime(back_dates).dt.strftime("%Y-%m-%d"),
        "Harga Aktual": back_actual,
        "Prediksi Model": back_pred,
        "Selisih (Rp)": back_pred - back_actual,
        "Selisih (%)": (back_pred - back_actual) / back_actual * 100,
    })
    back_df.index = range(1, len(back_df) + 1)
    back_df.index.name = "No"
    st.dataframe(
        back_df.style.format({
            "Harga Aktual": "Rp{:,.0f}",
            "Prediksi Model": "Rp{:,.0f}",
            "Selisih (Rp)": "{:+,.0f}",
            "Selisih (%)": "{:+.2f}%",
        }),
        use_container_width=True,
    )
    st.caption(
        f"Garis ungu putus-putus pada grafik = prediksi model untuk {n_back} hari bursa terakhir. "
        f"MAPE {n_back} hari ini: **{mape(back_actual, back_pred):.2f}%**. "
        "Tiap titik adalah prediksi satu langkah ke depan memakai data aktual hari-hari sebelumnya."
    )

    trend = "naik 📈" if future_preds[-1] > price_df["Close"].iloc[-1] else "turun 📉"
    st.info(f"Berdasarkan model, tren harga dalam {forecast_days} hari ke depan diperkirakan **{trend}** "
            f"dari Rp{price_df['Close'].iloc[-1]:,.0f} menjadi sekitar Rp{future_preds[-1]:,.0f}.")
    st.caption("⚠️ Prediksi model bersifat estimasi statistik dan **bukan rekomendasi investasi**.")

    is_naik = future_preds[-1] > price_df["Close"].iloc[-1]
    matching_sentiment = "Positif" if is_naik else "Negatif"

    st.subheader(f"💬 Komentar Sejalan dengan Tren Prediksi ({matching_sentiment})")
    st.caption(
        f"Karena model memprediksi harga akan **{trend}**, komentar bersentimen "
        f"**{matching_sentiment}** dianggap sejalan dengan arah prediksi tersebut. "
        f"Total komentar Positif dan Negatif ditampilkan di bawah."
    )

    min_comment_date = comments_df["PostDate"].min().date()
    max_comment_date = comments_df["PostDate"].max().date()

    col_dari_sejalan, col_sampai_sejalan = st.columns(2)
    with col_dari_sejalan:
        start_date = st.date_input(
            "🗓️ Dari tanggal",
            value=st.session_state.get("sejalan_start_value", max_comment_date),
            min_value=min_comment_date,
            max_value=max_comment_date,
            key="sejalan_start",
        )
    with col_sampai_sejalan:
        end_date = st.date_input(
            "🗓️ Sampai tanggal",
            value=st.session_state.get("sejalan_end_value", max_comment_date),
            min_value=min_comment_date,
            max_value=max_comment_date,
            key="sejalan_end",
        )

    st.session_state["sejalan_start_value"] = start_date
    st.session_state["sejalan_end_value"] = end_date

    if start_date > end_date:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
        start_date, end_date = end_date, start_date

    comments_scope = comments_df[
        (comments_df["PostDate"].dt.date >= start_date)
        & (comments_df["PostDate"].dt.date <= end_date)
    ]
    label_tanggal_sejalan = (
        f"**{start_date}**" if start_date == end_date
        else f"**{start_date}** – **{end_date}**"
    )
    st.caption(
        f"Menampilkan komentar pada {label_tanggal_sejalan} "
        f"({len(comments_scope)} dari {len(comments_df)} total komentar)."
    )

    pos_count = int((comments_scope["Sentiment"] == "Positif").sum())
    neg_count = int((comments_scope["Sentiment"] == "Negatif").sum())
    matching_count = pos_count if is_naik else neg_count
    total_comments = pos_count + neg_count
    matching_pct = (matching_count / total_comments * 100) if total_comments else 0.0

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric(
        "Total Komentar Positif", f"{pos_count}",
        help="Jumlah komentar bersentimen Positif pada rentang tanggal yang dipilih.",
    )
    mcol2.metric(
        "Total Komentar Negatif", f"{neg_count}",
        help="Jumlah komentar bersentimen Negatif pada rentang tanggal yang dipilih.",
    )
    mcol3.metric(
        f"Proporsi Sejalan ({matching_sentiment})", f"{matching_pct:.1f}%",
        help=f"Persentase komentar {matching_sentiment} dari total komentar Positif + Negatif "
             f"pada rentang tanggal yang dipilih — sentimen ini yang sejalan dengan prediksi "
             f"harga {trend}.",
    )

    st.markdown("**📰 Daftar Komentar Positif & Negatif**")

    semua_scope = comments_scope.sort_values("PostDate", ascending=False)
    if len(semua_scope):
        show_table(semua_scope[KOLOM_SENTIMEN], height=400)
        st.caption(
            f"Menampilkan seluruh {len(semua_scope)} komentar "
            f"({pos_count} Positif, {neg_count} Negatif). "
            f"Sentimen yang sejalan dengan prediksi: **{matching_sentiment}**."
        )
    else:
        st.caption("Belum ada komentar pada rentang tanggal ini.")


# ------------------------------------------------------------------
# HALAMAN: ANALISIS SENTIMEN KOMENTAR
# ------------------------------------------------------------------
elif page == "💬 Analisis Sentimen Komentar":
    st.title("💬 Analisis Sentimen Komentar")
    st.caption(
        "Metode: pencocokan kata dasar baku (KBBI) + stemmer imbuhan, dengan penanganan "
        "negasi, penguat, dan kata kontras (tetapi)."
    )

    color_map = {"Positif": "#2ca02c", "Negatif": "#d62728"}
    sentiment_counts = comments_df["Sentiment"].value_counts()

    c1, c2 = st.columns(2)
    c1.metric("Positif", int(sentiment_counts.get("Positif", 0)))
    c2.metric("Negatif", int(sentiment_counts.get("Negatif", 0)))

    # ----------------------------------------------------------------
    # EDA: DISTRIBUSI LABEL & PANJANG KOMENTAR
    # ----------------------------------------------------------------
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    tab_label, tab_len = st.tabs(["📊 Distribusi Label Sentimen", "📏 Panjang Komentar"])

    # ---------- TAB 1: DISTRIBUSI LABEL ----------
    with tab_label:
        label_df = pd.DataFrame({
            "Sentimen": sentiment_counts.index,
            "Jumlah": sentiment_counts.values,
        })
        label_df["Persentase"] = label_df["Jumlah"] / label_df["Jumlah"].sum() * 100
        label_df["Teks"] = label_df.apply(
            lambda r: f"{int(r['Jumlah'])} ({r['Persentase']:.1f}%)", axis=1
        )

        fig_label = px.bar(
            label_df, x="Sentimen", y="Jumlah", color="Sentimen", text="Teks",
            color_discrete_map=color_map,
            title="Jumlah Komentar per Label Sentimen",
        )
        fig_label.update_traces(textposition="outside")
        fig_label.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_label, use_container_width=True)

        if len(label_df) == 2 and label_df["Jumlah"].min() > 0:
            ratio = label_df["Jumlah"].max() / label_df["Jumlah"].min()
            kelas_mayor = label_df.loc[label_df["Jumlah"].idxmax(), "Sentimen"]
            st.caption(
                f"Kelas mayoritas: **{kelas_mayor}** · Rasio mayoritas : minoritas = **{ratio:.2f} : 1**."
            )
        st.caption(
            "Komentar seri atau tanpa kata sentimen tetap diberi label Positif/Negatif "
            "lewat aturan cadangan (kata sentimen terakhir, atau label default)."
        )

    # ---------- TAB 2: PANJANG KOMENTAR ----------
    with tab_len:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rata-rata Kata", f"{comments_df['JumlahKata'].mean():.1f}")
        m2.metric("Median Kata", f"{comments_df['JumlahKata'].median():.0f}")
        m3.metric("Terpendek (kata)", f"{int(comments_df['JumlahKata'].min())}")
        m4.metric("Terpanjang (kata)", f"{int(comments_df['JumlahKata'].max())}")

        satuan = st.radio(
            "Ukuran panjang komentar", ["Jumlah Kata", "Jumlah Karakter"],
            horizontal=True, key="eda_satuan",
        )
        kolom_len = "JumlahKata" if satuan == "Jumlah Kata" else "JumlahKarakter"

        fig_hist = px.histogram(
            comments_df, x=kolom_len, color="Sentiment", nbins=30,
            color_discrete_map=color_map, barmode="overlay", opacity=0.75,
            title=f"Distribusi {satuan} per Komentar",
        )
        fig_hist.update_layout(xaxis_title=satuan, yaxis_title="Jumlah Komentar", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**Statistik deskriptif panjang komentar per sentimen**")
        desc = (
            comments_df.groupby("Sentiment")[["JumlahKata", "JumlahKarakter"]]
            .agg(["count", "mean", "median", "min", "max", "std"])
            .round(2)
        )
        desc.columns = [f"{a} ({b})" for a, b in desc.columns]
        st.dataframe(desc, use_container_width=True)

        st.markdown("**Komentar terpanjang & terpendek**")
        ekstrem = pd.concat([
            comments_df.nlargest(3, "JumlahKata").assign(Kategori="Terpanjang"),
            comments_df.nsmallest(3, "JumlahKata").assign(Kategori="Terpendek"),
        ])[["Kategori", "Username", "TweetText", "JumlahKata", "Sentiment"]]
        show_table(ekstrem)

    st.subheader("Filter & Telusuri Komentar")
    filt = st.multiselect("Filter Sentimen", ["Positif", "Negatif"],
                          default=["Positif", "Negatif"])
    keyword = st.text_input("Cari kata kunci dalam komentar (opsional)")

    filtered = comments_df[comments_df["Sentiment"].isin(filt)]
    if keyword:
        filtered = filtered[filtered["TweetText"].str.contains(keyword, case=False, na=False)]

    show_table(
        filtered[KOLOM_SENTIMEN].sort_values("PostDate", ascending=False),
        height=400,
    )
    st.caption(f"Menampilkan {len(filtered)} dari {len(comments_df)} komentar.")

# ------------------------------------------------------------------
# HALAMAN: KESIMPULAN AKHIR (GABUNGAN PREDIKSI HARGA + SENTIMEN)
# ------------------------------------------------------------------
elif page == "📌 Kesimpulan Akhir":
    st.title("📌 Kesimpulan Akhir: Prediksi Harga + Sentimen Komentar")
    st.caption(
        "Ringkasan gabungan antara hasil model LSTM (prediksi harga) dan hasil "
        "analisis sentimen lexicon-based (komentar), untuk memudahkan membaca "
        "keduanya sekaligus dalam satu tampilan."
    )

    if "lstm_result" not in st.session_state:
        st.warning(
            "Model LSTM belum dilatih. Buka halaman **📈 Prediksi Harga (LSTM)**, "
            "atur parameter di sidebar, lalu klik **🚀 Latih Model & Prediksi** "
            "terlebih dahulu agar kesimpulan di halaman ini bisa dihitung."
        )
        st.stop()

    result = st.session_state["lstm_result"]
    forecast_days = st.session_state["lstm_params"]["forecast_days"]

    with st.spinner("Menyiapkan ringkasan..."):
        future_preds = forecast_future(result, forecast_days)

    last_actual_price = float(price_df["Close"].iloc[-1])
    last_pred_price = float(future_preds[-1])
    price_change = last_pred_price - last_actual_price
    price_change_pct = (price_change / last_actual_price) * 100 if last_actual_price else 0.0
    is_naik = last_pred_price > last_actual_price
    trend_label = "NAIK 📈" if is_naik else "TURUN 📉"

    rmse_test = rmse(result["y_test_actual"], result["test_pred"])
    mape_test = mape(result["y_test_actual"], result["test_pred"])

    pos_count = int((comments_df["Sentiment"] == "Positif").sum())
    neg_count = int((comments_df["Sentiment"] == "Negatif").sum())
    total_comments = pos_count + neg_count
    pos_pct = (pos_count / total_comments * 100) if total_comments else 0.0
    neg_pct = (neg_count / total_comments * 100) if total_comments else 0.0
    sentimen_dominan = "Positif" if pos_count >= neg_count else "Negatif"

    prediksi_arah = "Positif" if is_naik else "Negatif"
    sejalan = prediksi_arah == sentimen_dominan

    if sejalan and is_naik:
        sinyal, sinyal_warna, sinyal_desc = (
            "🟢 Optimis (Sejalan)",
            "success",
            "Prediksi model mengarah **naik** dan sentimen komentar didominasi "
            "**Positif** — kedua sinyal ini sejalan/mendukung arah yang sama.",
        )
    elif sejalan and not is_naik:
        sinyal, sinyal_warna, sinyal_desc = (
            "🔴 Pesimis (Sejalan)",
            "error",
            "Prediksi model mengarah **turun** dan sentimen komentar didominasi "
            "**Negatif** — kedua sinyal ini sejalan/mendukung arah yang sama.",
        )
    else:
        sinyal, sinyal_warna, sinyal_desc = (
            "🟡 Campuran (Tidak Sejalan)",
            "warning",
            f"Prediksi model mengarah **{'naik' if is_naik else 'turun'}**, tetapi "
            f"sentimen komentar justru didominasi **{sentimen_dominan}** — kedua "
            f"sinyal ini berlawanan arah, sehingga perlu dibaca dengan lebih hati-hati.",
        )

    st.subheader("🔎 Ringkasan Sinyal Gabungan")
    getattr(st, sinyal_warna)(f"**{sinyal}** — {sinyal_desc}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Prediksi Harga (akhir periode)",
        f"Rp{last_pred_price:,.0f}",
        f"{price_change:+,.0f} ({price_change_pct:+.2f}%)",
    )
    k2.metric("Arah Tren Prediksi", trend_label)
    k3.metric("Sentimen Dominan", f"{sentimen_dominan} ({max(pos_pct, neg_pct):.1f}%)")
    k4.metric("Akurasi Model (MAPE)", f"{mape_test:.2f}%")

    st.markdown("---")

    col_harga, col_sentimen = st.columns(2)

    with col_harga:
        st.markdown("### 📈 Hasil Prediksi Harga (LSTM)")
        st.write(f"- Harga terakhir (aktual): **Rp{last_actual_price:,.0f}**")
        st.write(
            f"- Prediksi harga {forecast_days} hari ke depan: **Rp{last_pred_price:,.0f}** "
            f"({price_change_pct:+.2f}%)"
        )
        st.write(f"- Arah tren: **{trend_label}**")
        st.write(f"- RMSE (data uji): **{rmse_test:,.2f}**")
        st.write(f"- MAPE (data uji): **{mape_test:.2f}%**")

    with col_sentimen:
        st.markdown("### 💬 Hasil Analisis Sentimen Komentar")
        st.write(f"- Total komentar dianalisis: **{total_comments}**")
        st.write(f"- Komentar Positif: **{pos_count}** ({pos_pct:.1f}%)")
        st.write(f"- Komentar Negatif: **{neg_count}** ({neg_pct:.1f}%)")
        st.write(f"- Sentimen dominan: **{sentimen_dominan}**")

    # ----------------------------------------------------------------
    # KOMENTAR YANG SEJALAN DENGAN ARAH PREDIKSI HARGA
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤝 Komentar yang Sejalan dengan Prediksi Harga")

    sentimen_sejalan = "Positif" if is_naik else "Negatif"
    sentimen_berlawanan = "Negatif" if is_naik else "Positif"

    comments_sejalan = comments_df[comments_df["Sentiment"] == sentimen_sejalan]
    comments_berlawanan = comments_df[comments_df["Sentiment"] == sentimen_berlawanan]

    # Kolom Kata Penentu & Keyakinan sengaja tidak ditampilkan di tabel maupun Excel
    kolom_komentar = ["Username", "PostDate", "Harga Saham (Close)", "TweetText",
                      "Sentiment"]

    jumlah_sejalan = len(comments_sejalan)
    jumlah_berlawanan = len(comments_berlawanan)
    pct_sejalan = (jumlah_sejalan / total_comments * 100) if total_comments else 0.0
    pct_berlawanan = (jumlah_berlawanan / total_comments * 100) if total_comments else 0.0

    st.info(
        f"Model memprediksi harga **{trend_label}**, sehingga komentar yang dianggap "
        f"**sejalan** adalah komentar **{sentimen_sejalan}** "
        f"({'saham naik → komentar positif' if is_naik else 'saham turun → komentar negatif'})."
    )

    s1, s2, s3 = st.columns(3)
    s1.metric(f"Komentar Sejalan ({sentimen_sejalan})", f"{jumlah_sejalan}", f"{pct_sejalan:.1f}%")
    s2.metric(
        f"Komentar Berlawanan ({sentimen_berlawanan})",
        f"{jumlah_berlawanan}",
        f"{pct_berlawanan:.1f}%",
        delta_color="off",
    )
    s3.metric("Total Komentar", f"{total_comments}")

    # ---------- FILTER TANGGAL (komentar & harga saham) ----------
    st.markdown("**🗓️ Filter Tanggal**")
    kes_min_date = min(price_df["Date"].min().date(), comments_df["PostDate"].min().date())
    kes_max_date = max(price_df["Date"].max().date(), comments_df["PostDate"].max().date())

    def _clamp(d):
        return max(kes_min_date, min(d, kes_max_date))

    fk1, fk2 = st.columns(2)
    with fk1:
        kes_start = st.date_input(
            "🗓️ Dari tanggal",
            value=_clamp(st.session_state.get("kes_start_value", kes_min_date)),
            min_value=kes_min_date,
            max_value=kes_max_date,
            key="kes_start",
        )
    with fk2:
        kes_end = st.date_input(
            "🗓️ Sampai tanggal",
            value=_clamp(st.session_state.get("kes_end_value", kes_max_date)),
            min_value=kes_min_date,
            max_value=kes_max_date,
            key="kes_end",
        )

    st.session_state["kes_start_value"] = kes_start
    st.session_state["kes_end_value"] = kes_end

    if kes_start > kes_end:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih besar dari tanggal 'Sampai'.")
        kes_start, kes_end = kes_end, kes_start

    label_tanggal_kes = (
        f"**{kes_start}**" if kes_start == kes_end
        else f"**{kes_start}** – **{kes_end}**"
    )

    komentar_filter = comments_df[
        (comments_df["PostDate"].dt.date >= kes_start)
        & (comments_df["PostDate"].dt.date <= kes_end)
    ]

    if len(komentar_filter):
        semua_komentar = add_close_price(komentar_filter, price_df)
        tampil_komentar = semua_komentar[kolom_komentar]
        show_table(tampil_komentar, height=400)
        pos_f = int((komentar_filter["Sentiment"] == "Positif").sum())
        neg_f = int((komentar_filter["Sentiment"] == "Negatif").sum())
        st.caption(
            f"Menampilkan {len(komentar_filter)} dari {len(comments_df)} komentar "
            f"pada {label_tanggal_kes} ({pos_f} Positif, {neg_f} Negatif). "
            f"Sentimen yang sejalan dengan prediksi harga {trend_label}: **{sentimen_sejalan}**."
        )

        excel_bytes = build_komentar_excel(tampil_komentar, sentimen_sejalan)
        st.download_button(
            "⬇️ Unduh Komentar + Hitungan Positif/Negatif (Excel)",
            data=excel_bytes,
            file_name=f"komentar_sejalan_{kes_start}_{kes_end}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_komentar_xlsx",
        )
    else:
        st.info(f"Tidak ada komentar pada tanggal {label_tanggal_kes}.")

    # ----------------------------------------------------------------
    # PREPROCESSING DATA TIME SERIES
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧹 Preprocessing Data Time Series (Harga Saham)")
    st.caption(
        "Tahapan pembersihan data harga: pengecekan missing value, penanganan "
        "missing value, dan penanganan outlier. Bagian ini bersifat informatif "
        "dan tidak mengubah data yang dipakai melatih model LSTM."
    )

    pp_missing = check_missing_values(price_df)
    pp_dup, pp_gap = check_time_gaps(price_df)
    pp_total_missing = int(pp_missing["Total Bermasalah"].sum())

    tab_cek, tab_isi, tab_out = st.tabs([
        "1️⃣ Pengecekan Missing Value",
        "2️⃣ Penanganan Missing Value",
        "3️⃣ Penanganan Outlier",
    ])

    # ---------- TAB 1: PENGECEKAN MISSING VALUE ----------
    with tab_cek:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Nilai Bermasalah", f"{pp_total_missing}")
        m2.metric("Tanggal Duplikat", f"{pp_dup}")
        m3.metric(
            "Hari Kerja Tanpa Data", f"{pp_gap}",
            help="Selisih hari kerja (Sen–Jum) pada rentang data. Umumnya hari libur bursa, "
                 "bukan data hilang, sehingga tidak diisi.",
        )
        st.dataframe(
            pp_missing.set_index("Kolom").style.format({"Persentase (%)": "{:.2f}%"}),
            use_container_width=True,
        )
        if pp_total_missing == 0 and pp_dup == 0:
            st.success("✅ Tidak ditemukan missing value, nilai 0, maupun tanggal duplikat pada data harga.")
        else:
            st.warning("⚠️ Ditemukan data bermasalah. Lihat tab *Penanganan Missing Value*.")

    # ---------- TAB 2: PENANGANAN MISSING VALUE ----------
    price_clean = handle_missing_values(price_df)
    with tab_isi:
        st.markdown(
            "**Metode:** baris dengan tanggal kosong/duplikat dibuang; harga ≤ 0 dianggap "
            "missing; seluruh missing diisi dengan **interpolasi berbasis waktu** "
            "(*forward/backward fill* untuk data di ujung deret). Hari libur bursa "
            "tidak diisi karena memang bukan hari perdagangan."
        )
        after_missing = check_missing_values(price_clean)
        after_total = int(after_missing["Total Bermasalah"].sum())
        a1, a2, a3 = st.columns(3)
        a1.metric("Bermasalah (Sebelum)", f"{pp_total_missing}")
        a2.metric(
            "Bermasalah (Sesudah)", f"{after_total}",
            delta=f"{after_total - pp_total_missing:+d}", delta_color="inverse",
        )
        a3.metric(
            "Jumlah Baris", f"{len(price_clean)}",
            delta=f"{len(price_clean) - len(price_df):+d}", delta_color="off",
        )
        st.dataframe(
            after_missing.set_index("Kolom").style.format({"Persentase (%)": "{:.2f}%"}),
            use_container_width=True,
        )

    # ---------- TAB 3: PENANGANAN OUTLIER ----------
    with tab_out:
        oc1, oc2 = st.columns(2)
        with oc1:
            out_method = st.radio(
                "Metode penanganan outlier",
                ["Interpolasi linear", "Median bergulir (window 5)", "Tidak ditangani"],
                key="pp_out_method",
            )
        with oc2:
            out_k = st.slider(
                "Pengali IQR (k)", 1.0, 3.0, 1.5, step=0.5, key="pp_out_k",
                help="Makin kecil k, makin banyak titik dianggap outlier.",
            )

        out_mask, out_ret, out_low, out_up = detect_outliers_iqr(price_clean, out_k)
        price_final = handle_outliers(price_clean, out_mask, out_method)
        n_out = int(out_mask.sum())

        o1, o2, o3 = st.columns(3)
        o1.metric(
            "Jumlah Outlier", f"{n_out}",
            f"{n_out / len(price_clean) * 100:.2f}% data", delta_color="off",
        )
        o2.metric("Batas Bawah Return", f"{out_low * 100:.2f}%")
        o3.metric("Batas Atas Return", f"{out_up * 100:.2f}%")

        fig_out = go.Figure()
        fig_out.add_trace(go.Scatter(
            x=price_clean["Date"], y=price_clean["Close"],
            name="Close (sebelum)", line=dict(color="lightgray"),
        ))
        fig_out.add_trace(go.Scatter(
            x=price_final["Date"], y=price_final["Close"],
            name="Close (sesudah)", line=dict(color="#2ca02c", width=1.5),
        ))
        if n_out:
            fig_out.add_trace(go.Scatter(
                x=price_clean.loc[out_mask, "Date"], y=price_clean.loc[out_mask, "Close"],
                mode="markers", name="Outlier",
                marker=dict(color="#d62728", size=9, symbol="x"),
            ))
        fig_out.update_layout(
            title="Deteksi & Penanganan Outlier pada Harga Close",
            xaxis_title="Tanggal", yaxis_title="Harga (Rp)", height=420,
        )
        st.plotly_chart(fig_out, use_container_width=True)

        if n_out:
            out_table = pd.DataFrame({
                "Tanggal": price_clean.loc[out_mask, "Date"].dt.strftime("%Y-%m-%d"),
                "Return Harian (%)": out_ret[out_mask] * 100,
                "Close Sebelum": price_clean.loc[out_mask, "Close"],
                "Close Sesudah": price_final.loc[out_mask, "Close"],
            })
            out_table.index = range(1, len(out_table) + 1)
            out_table.index.name = "No"
            st.dataframe(
                out_table.style.format({
                    "Return Harian (%)": "{:+.2f}%",
                    "Close Sebelum": "Rp{:,.0f}",
                    "Close Sesudah": "Rp{:,.0f}",
                }),
                use_container_width=True,
            )
        else:
            st.success("✅ Tidak ada outlier terdeteksi pada pengali IQR ini.")
        st.caption(
            "Outlier dihitung dari return harian Close dengan aturan IQR. Lonjakan harga bisa saja "
            "akibat peristiwa nyata (mis. aksi korporasi), jadi tinjau tabel di atas sebelum memutuskan "
            "memakai data hasil penanganan."
        )

    st.markdown("---")

    st.subheader("📋 Tabel Ringkasan Akhir")
    summary_df = pd.DataFrame({
        "Metrik": [
            "Harga Terakhir (Aktual)",
            f"Prediksi Harga ({forecast_days} Hari ke Depan)",
            "Perubahan Harga (Rp)",
            "Perubahan Harga (%)",
            "Arah Tren Prediksi",
            "RMSE (Data Uji)",
            "MAE (Data Uji)",
            "MAPE (Data Uji)",
            "Missing Value / Nilai 0 (Sebelum Penanganan)",
            "Outlier Terdeteksi (IQR Return)",
            "Total Komentar",
            "Komentar Positif",
            "Komentar Negatif",
            "Sentimen Dominan",
            "Sinyal Gabungan",
            "Sentimen Sejalan dengan Prediksi",
            "Jumlah Komentar Sejalan",
        ],
        "Nilai": [
            f"Rp{last_actual_price:,.0f}",
            f"Rp{last_pred_price:,.0f}",
            f"{price_change:+,.0f}",
            f"{price_change_pct:+.2f}%",
            trend_label,
            f"{rmse_test:,.2f}",
            f"{mean_absolute_error(result['y_test_actual'], result['test_pred']):,.2f}",
            f"{mape_test:.2f}%",
            f"{pp_total_missing}",
            f"{n_out} ({out_method})",
            f"{total_comments}",
            f"{pos_count} ({pos_pct:.1f}%)",
            f"{neg_count} ({neg_pct:.1f}%)",
            sentimen_dominan,
            sinyal,
            sentimen_sejalan,
            f"{jumlah_sejalan} ({pct_sejalan:.1f}%)",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Unduh Ringkasan (CSV)",
        data=csv_bytes,
        file_name="kesimpulan_akhir_bbri.csv",
        mime="text/csv",
    )

    st.caption(
        "⚠️ Ringkasan ini digabungkan dari estimasi statistik model LSTM dan "
        "analisis sentimen berbasis kamus kata sederhana — **bukan rekomendasi "
        "investasi**. Selalu lakukan riset & pertimbangan risiko sendiri sebelum "
        "mengambil keputusan trading/investasi."
    )