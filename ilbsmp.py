"""
ILBSMP - Improved Learning-Based Stock Market Prediction
=========================================================
Implementation of the pipeline from Gokaraju & Vara Prasad,
"Stock market prediction using IALSTM-BIGRU architecture and improved
chi-square based selective feature set" (Statistics, 2026):

  1. Min-max normalization (fitted on the TRAINING rows only - no look-ahead)
  2. Technical features: EMA, SMA, DEMA, WWS, MACD, RSI, DMI (+DI, -DI, ADX)
  3. Improved chi-square feature selection (chi-square x Shannon entropy)
  4. IA-LSTM (LSTM + clipped attention + Leaky Bipolar activation) and Bi-GRU
  5. Improved score-level fusion (tanh normalization + Lehmer mean + confidence weights)

Notes on interpretation (the paper's equations 11 and 26-30 are typeset
ambiguously, so these are explicit, documented choices):
  * Eq. 11 -> chi-square statistic multiplied by the feature's normalised
    Shannon entropy, so features must be both associated with the target AND
    informative (non-degenerate) to rank highly.
  * Eq. 26-30 -> the Lehmer mean sum(x^p)/sum(x^(p-1)) is exactly a weighted
    mean with weights proportional to x^(p-1). Fusion therefore weights each
    model by (confidence share) x (tanh-normalised score)^(p-1) and returns the
    weighted average of the predictions, which keeps the output in price units.

TensorFlow is imported lazily, so everything except model training works
without it installed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "Close", "Volume", "EMA", "SMA", "DEMA", "WWS",
    "MACD", "MACD_signal", "RSI", "+DI", "-DI", "ADX",
]
WARMUP_ROWS = 30  # drop the first rows so slow indicators have settled


# ----------------------------------------------------------------------------
# Technical indicators
# ----------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Welles Wilder's smoothing (WWS): EMA with alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def dema(s: pd.Series, n: int) -> pd.Series:
    e = ema(s, n)
    return 2 * e - ema(e, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(loss != 0, 100.0)


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Directional Movement Index: returns (+DI, -DI, ADX)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = wilder(tr, n).replace(0, np.nan)
    plus_di = 100 * wilder(plus_dm, n) / atr
    minus_di = 100 * wilder(minus_dm, n) / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = wilder(dx.fillna(0.0), n)
    return plus_di, minus_di, adx


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The 12 candidate features the feature-selection step chooses from."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["Close"] = c
    f["Volume"] = v
    f["EMA"] = ema(c, 20)
    f["SMA"] = sma(c, 20)
    f["DEMA"] = dema(c, 20)
    f["WWS"] = wilder(c, 14)
    f["MACD"] = ema(c, 12) - ema(c, 26)
    f["MACD_signal"] = ema(f["MACD"], 9)
    f["RSI"] = rsi(c, 14)
    plus_di, minus_di, adx = dmi(h, l, c, 14)
    f["+DI"], f["-DI"], f["ADX"] = plus_di, minus_di, adx
    return f[FEATURE_NAMES]


# ----------------------------------------------------------------------------
# Improved chi-square feature selection (chi-square x Shannon entropy)
# ----------------------------------------------------------------------------
def _discretize(x, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.digitize(x, edges)


def shannon_entropy(codes: np.ndarray) -> float:
    counts = np.bincount(codes)
    p = counts[counts > 0] / counts.sum()
    return float(max(0.0, -(p * np.log2(p)).sum()))


def chi_square_stat(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    table = pd.crosstab(x_codes, y_codes).values.astype(float)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    n = table.sum()
    expected = row @ col / n
    return float(((table - expected) ** 2 / expected).sum())


def improved_chi_square_scores(X: pd.DataFrame, y, bins: int = 10) -> pd.DataFrame:
    """Rank features by chi-square(feature, target) x normalised Shannon entropy."""
    y_codes = _discretize(y, bins)
    rows = []
    for col in X.columns:
        x_codes = _discretize(X[col].values, bins)
        chi2 = chi_square_stat(x_codes, y_codes)
        h_norm = shannon_entropy(x_codes) / np.log2(bins)
        rows.append({"feature": col, "chi2": chi2, "entropy": h_norm, "score": chi2 * h_norm})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# Leaky Bipolar activation (Leaky ReLU + bipolar sigmoid), Eq. 19-21
# ----------------------------------------------------------------------------
def leaky_bipolar_np(x, k: float = 0.1):
    x = np.asarray(x, dtype=float)
    bipolar = (1 - np.exp(-k * x)) / (1 + np.exp(-k * x))
    return bipolar + np.maximum(k * x, x)


# ----------------------------------------------------------------------------
# Improved score-level fusion (tanh normalisation + Lehmer mean + confidence)
# ----------------------------------------------------------------------------
def tanh_normalize(x, ref=None, eps: float = 1e-12) -> np.ndarray:
    """Tanh estimator: 0.5 * (tanh(0.01 * (x - mean) / sd) + 1), stats from `ref`."""
    x = np.asarray(x, dtype=float)
    ref = x if ref is None else np.asarray(ref, dtype=float)
    mu, sd = ref.mean(), ref.std() + eps
    return 0.5 * (np.tanh(0.01 * (x - mu) / sd) + 1.0)


def confidence_from_errors(errors, lo: float = 0.6, hi: float = 0.8) -> np.ndarray:
    """Map validation errors to confidence in [lo, hi]: lowest error -> hi."""
    e = np.asarray(errors, dtype=float)
    span = e.max() - e.min()
    if span <= 1e-15:
        return np.full_like(e, hi)
    return hi - (hi - lo) * (e - e.min()) / span


def fuse_predictions(preds, confidences, p: float = 2.0, refs=None):
    """
    preds       : list of 1-D arrays (one per model), same length
    confidences : per-model confidence in [alpha, beta]
    p           : Lehmer-mean power
    refs        : optional reference series per model for the tanh statistics
                  (used to fuse a single new point consistently)
    Returns (fused, W, a) where a[k, t] are the effective per-step weights.
    """
    preds = [np.asarray(x, dtype=float) for x in preds]
    refs = preds if refs is None else [np.asarray(r, dtype=float) for r in refs]
    conf = np.asarray(confidences, dtype=float)
    W = conf / conf.sum()  # Eq. 28
    scores = np.vstack([tanh_normalize(pr, rf) for pr, rf in zip(preds, refs)])  # (m, T)
    a = W[:, None] * scores ** (p - 1.0)  # Lehmer emphasis on higher-scoring predictions
    a = a / a.sum(axis=0, keepdims=True)
    fused = (a * np.vstack(preds)).sum(axis=0)
    return fused, W, a


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    safe_true = np.where(y_true == 0, np.nan, y_true)
    return {
        "MSE": mse,
        "MAE": float(np.mean(np.abs(err))),
        "MSLE": float(np.mean((np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(y_pred, 0, None))) ** 2)),
        "MAPE (%)": float(np.nanmean(np.abs(err / safe_true)) * 100),
        "RMSE": float(np.sqrt(mse)),
    }


# ----------------------------------------------------------------------------
# Dataset preparation (no TensorFlow needed)
# ----------------------------------------------------------------------------
def prepare_dataset(
    df: pd.DataFrame,
    train_pct: int = 80,
    lookback: int = 10,
    n_features: int = 8,
    bins: int = 10,
    target_mode: str = "price",
) -> dict:
    """
    target_mode: "price"  -> predict next-day Close (as in the paper)
                 "return" -> predict next-day return, converted back to a price
    Feature selection and min-max scaling use ONLY rows that belong to the
    training period, so nothing from the test period leaks into the model.
    """
    feats = build_feature_frame(df).replace([np.inf, -np.inf], np.nan).dropna().iloc[WARMUP_ROWS:]
    close = feats["Close"]
    n_rows = len(feats)
    if n_rows < lookback + 150:
        raise ValueError(
            f"Only {n_rows} usable rows after indicator warm-up - choose a longer training history "
            f"or a shorter look-back window."
        )

    next_close = close.shift(-1)
    target = next_close / close - 1.0 if target_mode == "return" else next_close

    n_seq = n_rows - lookback  # sequences whose next-day target exists
    n_train = int(n_seq * train_pct / 100)
    if n_train < 100 or n_seq - n_train < 20:
        raise ValueError("Not enough rows for this train/test split - use a longer history or a smaller split.")
    last_train_end = n_train + lookback - 2  # row index where the last training target is known
    fit_rows = slice(0, last_train_end + 1)

    table = improved_chi_square_scores(feats.iloc[fit_rows], target.iloc[fit_rows], bins)
    table["selected"] = False
    table.loc[: max(1, min(n_features, len(table))) - 1, "selected"] = True
    selected = table.loc[table["selected"], "feature"].tolist()

    xv = feats[selected].values.astype(float)
    x_min = xv[fit_rows].min(axis=0)
    x_rng = xv[fit_rows].max(axis=0) - x_min
    x_rng = np.where(x_rng == 0, 1.0, x_rng)
    xs = (xv - x_min) / x_rng

    yv = target.values.astype(float)
    y_min = float(np.nanmin(yv[fit_rows]))
    y_rng = float(np.nanmax(yv[fit_rows]) - y_min) or 1.0
    ys = (yv - y_min) / y_rng

    idx_end = np.arange(lookback - 1, n_rows - 1)
    x_seq = np.stack([xs[i - lookback + 1 : i + 1] for i in idx_end]).astype(np.float32)
    y_seq = ys[idx_end].astype(np.float32)
    x_last = xs[n_rows - lookback : n_rows][None, :, :].astype(np.float32)

    n_fit = int(n_train * 0.85)
    return {
        "feature_table": table,
        "selected": selected,
        "x_fit": x_seq[:n_fit], "y_fit": y_seq[:n_fit],
        "x_val": x_seq[n_fit:n_train], "y_val": y_seq[n_fit:n_train],
        "x_test": x_seq[n_train:], "y_test": y_seq[n_train:],
        "x_last": x_last,
        "close_end_test": close.values[idx_end][n_train:],
        "actual_next_test": next_close.values[idx_end][n_train:],
        "test_dates": feats.index[idx_end + 1][n_train:],
        "last_close": float(close.iloc[-1]),
        "last_date": feats.index[-1],
        "y_min": y_min, "y_rng": y_rng, "target_mode": target_mode,
        "n_rows": n_rows, "n_train": n_train, "n_test": n_seq - n_train,
    }


def to_price(ds: dict, y_scaled, close_end) -> np.ndarray:
    raw = np.asarray(y_scaled, dtype=float) * ds["y_rng"] + ds["y_min"]
    if ds["target_mode"] == "return":
        return np.asarray(close_end, dtype=float) * (1.0 + raw)
    return raw


# ----------------------------------------------------------------------------
# Models (TensorFlow / Keras 3 - imported lazily)
# ----------------------------------------------------------------------------
def tensorflow_available() -> bool:
    """Cheap check (does not import TensorFlow, which is slow and memory-hungry)."""
    from importlib.util import find_spec

    try:
        return find_spec("tensorflow") is not None
    except Exception:
        return False


def build_models(lookback: int, n_features: int, seed: int = 42):
    """Returns (ia_lstm, bi_gru) Keras models. Requires TensorFlow >= 2.16 (Keras 3)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    ops = keras.ops
    keras.utils.set_random_seed(seed)

    class IAAttention(keras.layers.Layer):
        """
        Improved attention: tanh scoring -> clipping layer (Eq. 18) ->
        Leaky Bipolar activation (Fig. 3, replaces softmax) -> weights normalised
        to sum(|a|) = 1 -> weighted sum of the LSTM hidden states.
        """

        def __init__(self, clip_min=-3.0, clip_max=3.0, k=0.1, **kwargs):
            super().__init__(**kwargs)
            self.clip_min, self.clip_max, self.k = clip_min, clip_max, k

        def build(self, input_shape):
            d = int(input_shape[-1])
            self.w = self.add_weight(name="w", shape=(d, d), initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="b", shape=(d,), initializer="zeros", trainable=True)
            self.u = self.add_weight(name="u", shape=(d, 1), initializer="glorot_uniform", trainable=True)
            super().build(input_shape)

        def call(self, h):  # h: (batch, time, units)
            e = ops.tanh(ops.einsum("btd,de->bte", h, self.w) + self.b)
            s = ops.einsum("bte,eo->bto", e, self.u)  # (batch, time, 1)
            s = ops.clip(s, self.clip_min, self.clip_max)  # clipping layer
            ex = ops.exp(-self.k * s)
            a = (1.0 - ex) / (1.0 + ex) + ops.maximum(self.k * s, s)  # Leaky Bipolar
            a = a / (ops.sum(ops.abs(a), axis=1, keepdims=True) + 1e-8)
            return ops.sum(a * h, axis=1)  # context vector (batch, units)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"clip_min": self.clip_min, "clip_max": self.clip_max, "k": self.k})
            return cfg

    # IA-LSTM: LSTM(128) -> Dropout(0.2) -> attention -> Dense(1)   (Table 2 / Table 4)
    inp = keras.Input(shape=(lookback, n_features))
    x = keras.layers.LSTM(128, return_sequences=True)(inp)
    x = keras.layers.Dropout(0.2)(x)
    x = IAAttention()(x)
    ia_lstm = keras.Model(inp, keras.layers.Dense(1)(x), name="IA_LSTM")

    # Bi-GRU: Bidirectional GRU(50) -> Dropout(0.2) -> Dense(1)   (Table 2 / Table 4)
    inp2 = keras.Input(shape=(lookback, n_features))
    y = keras.layers.Bidirectional(keras.layers.GRU(50))(inp2)
    y = keras.layers.Dropout(0.2)(y)
    bi_gru = keras.Model(inp2, keras.layers.Dense(1)(y), name="Bi_GRU")

    for m in (ia_lstm, bi_gru):
        m.compile(optimizer="adam", loss="mean_squared_error")
    return ia_lstm, bi_gru


def train_models(ds: dict, epochs: int = 50, batch_size: int = 64, seed: int = 42, progress_cb=None) -> dict:
    """Train both models. progress_cb(fraction, message) is optional."""
    from tensorflow import keras

    lookback, n_feat = ds["x_fit"].shape[1], ds["x_fit"].shape[2]
    ia_lstm, bi_gru = build_models(lookback, n_feat, seed)

    out = {}
    for stage, (name, model) in enumerate([("IA-LSTM", ia_lstm), ("Bi-GRU", bi_gru)]):

        class _Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None, _stage=stage, _name=name):
                if progress_cb:
                    frac = (_stage + (epoch + 1) / epochs) / 2.0
                    progress_cb(frac, f"Training {_name}: epoch {epoch + 1}/{epochs}")

        hist = model.fit(
            ds["x_fit"], ds["y_fit"],
            validation_data=(ds["x_val"], ds["y_val"]),
            epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[_Progress()],
        )
        out[name] = {
            "val": model.predict(ds["x_val"], verbose=0).reshape(-1),
            "test": model.predict(ds["x_test"], verbose=0).reshape(-1),
            "last": float(model.predict(ds["x_last"], verbose=0).reshape(-1)[0]),
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        }
    return out


# ----------------------------------------------------------------------------
# End-to-end orchestration
# ----------------------------------------------------------------------------
def run_ilbsmp(
    df: pd.DataFrame,
    train_pct: int = 80,
    lookback: int = 10,
    n_features: int = 8,
    epochs: int = 50,
    target_mode: str = "price",
    conf_lo: float = 0.6,
    conf_hi: float = 0.8,
    lehmer_p: float = 2.0,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    ds = prepare_dataset(df, train_pct, lookback, n_features, target_mode=target_mode)
    trained = train_models(ds, epochs=epochs, seed=seed, progress_cb=progress_cb)

    names = ["IA-LSTM", "Bi-GRU"]
    test_price = {n: to_price(ds, trained[n]["test"], ds["close_end_test"]) for n in names}
    last_price_pred = {
        n: float(to_price(ds, [trained[n]["last"]], [ds["last_close"]])[0]) for n in names
    }

    # Confidence from validation error (validation slice was never trained on)
    val_err = [float(np.mean((trained[n]["val"] - ds["y_val"]) ** 2)) for n in names]
    confidence = confidence_from_errors(val_err, conf_lo, conf_hi)

    refs = [test_price[n] for n in names]
    fused_test, weights, _ = fuse_predictions(refs, confidence, p=lehmer_p)
    fused_next, _, _ = fuse_predictions(
        [np.array([last_price_pred[n]]) for n in names], confidence, p=lehmer_p, refs=refs
    )

    y_true = ds["actual_next_test"]
    metrics = {
        "IA-LSTM": regression_metrics(y_true, test_price["IA-LSTM"]),
        "Bi-GRU": regression_metrics(y_true, test_price["Bi-GRU"]),
        "Fused (ILBSMP)": regression_metrics(y_true, fused_test),
        "Naive (yesterday's close)": regression_metrics(y_true, ds["close_end_test"]),
    }

    return {
        "feature_table": ds["feature_table"],
        "selected": ds["selected"],
        "test_dates": ds["test_dates"],
        "y_true": y_true,
        "pred_ia": test_price["IA-LSTM"],
        "pred_bigru": test_price["Bi-GRU"],
        "pred_fused": fused_test,
        "metrics": pd.DataFrame(metrics).T,
        "confidence": dict(zip(names, map(float, confidence))),
        "weights": dict(zip(names, map(float, weights))),
        "val_mse_scaled": dict(zip(names, val_err)),
        "next_pred": {
            "IA-LSTM": last_price_pred["IA-LSTM"],
            "Bi-GRU": last_price_pred["Bi-GRU"],
            "Fused (ILBSMP)": float(fused_next[0]),
        },
        "last_close": ds["last_close"],
        "last_date": ds["last_date"],
        "history": {n: trained[n]["history"] for n in names},
        "n_rows": ds["n_rows"], "n_train": ds["n_train"], "n_test": ds["n_test"],
        "target_mode": target_mode,
    }
