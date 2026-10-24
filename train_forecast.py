#!/usr/bin/env python3
"""
PM2.5 forecaster for Thai provinces (Open-Meteo data) -> forecast.json

What it does
  1. Downloads hourly PM2.5 history (+ weather) per province from Open-Meteo.
  2. Cleans it (duplicates, gaps, impossible values) and builds features that
     only use information available at the forecast origin (no future leakage).
  3. Trains, for each horizon (3/6/12/24 h): Ridge and Gradient Boosting, plus
     10%/90% quantile models for a prediction interval.
  4. Evaluates honestly on the most recent days, never used for training, and
     compares against two simple baselines (persistence, climatology).
  5. Forecasts from the latest data and writes forecast.json (atomically).

Usage
  python train_forecast.py --selftest        # offline test on synthetic data
  python train_forecast.py --out forecast.json
  python train_forecast.py --provinces kanchanaburi --cache-dir .cache -v

Data caveat: Open-Meteo PM2.5 is itself a computer-model (CAMS) product, not a
ground sensor. This script measures how well we reproduce/forecast THAT series.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("pm25")

# ----------------------------------------------------------------- config ---
TZ = "Asia/Bangkok"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_START = date(2022, 8, 1)          # global air-quality data starts ~2022-07-29
ARCHIVE_LAG_DAYS = 6                   # archive weather lags a few days behind
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "precipitation", "surface_pressure"]
HORIZONS = [3, 6, 12, 24]
PROVINCES = {
    "kanchanaburi": {"name": "กาญจนบุรี", "lat": 14.02, "lon": 99.53},
    "chiangmai":    {"name": "เชียงใหม่",  "lat": 18.79, "lon": 98.98},
    "bangkok":      {"name": "กรุงเทพฯ",   "lat": 13.76, "lon": 100.50},
}
# Thai PCD 24h PM2.5 levels (ug/m3): upper edges of the first four levels
EDGES = np.array([15.0, 25.0, 37.5, 75.0])
LEVELS = ["ดีมาก", "ดี", "ปานกลาง", "เริ่มมีผลต่อสุขภาพ", "มีผลต่อสุขภาพ"]
ALERT = 37.5                           # above this: "starts to affect health"
PM_MAX = 1000.0                        # anything above is a sensor/model glitch
MIN_TRAIN_ROWS, MIN_TEST_ROWS = 2000, 500


class FatalAPIError(Exception):
    """4xx from the API (bad request): retrying will not help."""


def level_of(v: float) -> str:
    return LEVELS[int(np.digitize(v, EDGES, right=True))]


# ------------------------------------------------------------ HTTP / data ---
def http_get_json(url: str, params: dict, retries: int = 5, timeout: int = 60) -> dict:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code >= 400:
                raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except FatalAPIError:
            raise
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"request failed after {retries} tries: {exc}") from exc
            LOG.warning("attempt %d/%d failed (%s); retrying in %.0fs", attempt, retries, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def hourly_to_df(payload: dict, columns: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"response has no hourly data: {str(payload)[:200]}")
    missing = [c for c in columns if c not in hourly]
    if missing:
        raise ValueError(f"response is missing columns {missing}")
    df = pd.DataFrame({k: hourly[k] for k in ["time"] + columns})
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_range(url: str, lat: float, lon: float, start: date, end: date,
                variables: list[str], chunk_days: int = 90) -> pd.DataFrame:
    frames, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        payload = http_get_json(url, {
            "latitude": lat, "longitude": lon, "hourly": ",".join(variables),
            "start_date": cur.isoformat(), "end_date": stop.isoformat(), "timezone": TZ})
        frames.append(hourly_to_df(payload, variables))
        cur = stop + timedelta(days=1)
        time.sleep(0.3)  # be polite to the free API
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_recent(url: str, lat: float, lon: float, variables: list[str], past_days: int = 14) -> pd.DataFrame:
    payload = http_get_json(url, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(variables),
        "past_days": past_days, "forecast_days": 1, "timezone": TZ})
    return hourly_to_df(payload, variables)


def now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")


class OpenMeteoSource:
    """Real data. history() for training, recent() for the live forecast."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def history(self, key: str, cfg: dict) -> pd.DataFrame:
        end = (now_local() - pd.Timedelta(days=ARCHIVE_LAG_DAYS)).date()
        start, cached = AIR_START, None
        path = self.cache_dir / f"{key}.csv" if self.cache_dir else None
        if path and path.exists():
            try:
                cached = pd.read_csv(path, index_col=0, parse_dates=True)
                start = max(AIR_START, (cached.index.max() - pd.Timedelta(days=3)).date())
                LOG.info("%s: cache found, refreshing from %s", key, start)
            except Exception as exc:  # corrupt cache -> start over
                LOG.warning("%s: ignoring bad cache (%s)", key, exc)
                cached = None
        air = fetch_range(AIR_URL, cfg["lat"], cfg["lon"], start, end, ["pm2_5"])
        wx = fetch_range(ARCHIVE_URL, cfg["lat"], cfg["lon"], start, end, WEATHER_VARS)
        df = air.join(wx, how="outer")
        if cached is not None:
            df = pd.concat([cached, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
        return df

    def recent(self, key: str, cfg: dict) -> pd.DataFrame:
        air = fetch_recent(AIR_URL, cfg["lat"], cfg["lon"], ["pm2_5"])
        wx = fetch_recent(FORECAST_URL, cfg["lat"], cfg["lon"], WEATHER_VARS)
        return air.join(wx, how="outer")


class SyntheticSource:
    """Offline fake data with realistic structure, used by --selftest only."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _make(self, key: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + sum(map(ord, key)))
        idx = pd.date_range(AIR_START, now_local(), freq="h")
        n = len(idx)
        doy = idx.dayofyear.values
        hr = idx.hour.values
        season = 25 * np.exp(-((doy - 60) / 35.0) ** 2) + 5
        diurnal = 6 * np.sin((hr - 4) / 24 * 2 * np.pi)
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 2.2)
        pm = np.clip(season + diurnal + ar, 1, None)
        temp = 27 + 4 * np.sin((hr - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.8, n)
        df = pd.DataFrame({
            "pm2_5": pm,
            "temperature_2m": temp,
            "relative_humidity_2m": np.clip(95 - 2.2 * (temp - 24) + rng.normal(0, 5, n), 15, 100),
            "wind_speed_10m": np.abs(rng.normal(8, 3, n)),
            "precipitation": np.where(rng.random(n) < 0.04, rng.exponential(2, n), 0.0),
            "surface_pressure": 1009 + rng.normal(0, 2, n).cumsum() * 0.02,
        }, index=idx)
        return df

    def history(self, key, cfg):
        df = self._make(key)
        return df[df.index <= now_local() - pd.Timedelta(days=ARCHIVE_LAG_DAYS)]

    def recent(self, key, cfg):
        df = self._make(key)
        return df[df.index >= now_local() - pd.Timedelta(days=14)]


# --------------------------------------------------------------- cleaning ---
def clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Regular hourly index, impossible values -> NaN, short gaps interpolated."""
    needed = ["pm2_5"] + WEATHER_VARS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    df = df[needed].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.loc[(df["pm2_5"] < 0) | (df["pm2_5"] > PM_MAX), "pm2_5"] = np.nan
    df.loc[(df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100), "relative_humidity_2m"] = np.nan
    df.loc[df["precipitation"] < 0, "precipitation"] = np.nan
    before = float(df["pm2_5"].isna().mean())
    df = df.interpolate(method="time", limit=6, limit_area="inside")
    after = float(df["pm2_5"].isna().mean())
    LOG.info("%s: %d hourly rows, PM2.5 missing %.1f%% -> %.1f%% after filling gaps <=6h",
             label, len(df), 100 * before, 100 * after)
    return df


# --------------------------------------------------------------- features ---
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything here uses data at or BEFORE the origin hour only."""
    pm = df["pm2_5"]
    f = pd.DataFrame(index=df.index)
    for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72, 168):
        f[f"pm_lag{lag}"] = pm.shift(lag)
    for w in (6, 24, 72, 168):
        f[f"pm_mean{w}"] = pm.rolling(w, min_periods=int(w * 0.8)).mean()
    f["pm_std24"] = pm.rolling(24, min_periods=20).std()
    f["pm_max24"] = pm.rolling(24, min_periods=20).max()
    f["pm_trend6"] = pm - pm.shift(6)
    for c in WEATHER_VARS:
        f[c] = df[c]
        f[f"{c}_mean24"] = df[c].rolling(24, min_periods=20).mean()
    f["precip_sum24"] = df["precipitation"].rolling(24, min_periods=20).sum()
    ang_h = 2 * np.pi * f.index.hour / 24
    ang_d = 2 * np.pi * f.index.dayofyear / 365.25
    f["origin_hour_sin"], f["origin_hour_cos"] = np.sin(ang_h), np.cos(ang_h)
    f["doy_sin"], f["doy_cos"] = np.sin(ang_d), np.cos(ang_d)
    return f


def add_target_features(base: pd.DataFrame, h: int) -> pd.DataFrame:
    X = base.copy()
    t = X.index + pd.Timedelta(hours=h)
    X["target_hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
    X["target_hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
    return X


# ------------------------------------------------------------------ models ---
def fit_point(kind: str, X: pd.DataFrame, y: pd.Series):
    if kind == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif kind == "hgb":
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)
    else:
        raise ValueError(kind)
    return m.fit(X, y)


def fit_quantile(X: pd.DataFrame, y: pd.Series, q: float):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.06,
                                      max_leaf_nodes=31, min_samples_leaf=40, early_stopping=False,
                                      random_state=0)
    return m.fit(X, y)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0, PM_MAX)


def climatology(y_tr: pd.Series, t_tr: pd.DatetimeIndex, t_new: pd.DatetimeIndex) -> np.ndarray:
    d = pd.DataFrame({"m": t_tr.month, "h": t_tr.hour, "y": y_tr.values})
    table = d.groupby(["m", "h"])["y"].mean()
    key = pd.MultiIndex.from_arrays([t_new.month, t_new.hour])
    return table.reindex(key).fillna(float(y_tr.mean())).values


def evaluate(y: np.ndarray, pred: np.ndarray, persist: np.ndarray) -> dict:
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    mae_p = float(np.mean(np.abs(persist - y)))
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y - y.mean()) ** 2))
    act, prd = y > ALERT, pred > ALERT
    tp, fp, fn = int(np.sum(act & prd)), int(np.sum(~act & prd)), int(np.sum(act & ~prd))
    return {
        "n": int(len(y)), "mae": mae, "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None, "bias": float(np.mean(err)),
        "skill_vs_persistence": (1 - mae / mae_p) if mae_p > 0 else None,
        "level_accuracy": float(np.mean(np.digitize(y, EDGES, right=True) == np.digitize(pred, EDGES, right=True))),
        "alert_events": int(act.sum()),
        "alert_recall": tp / (tp + fn) if (tp + fn) else None,
        "alert_precision": tp / (tp + fp) if (tp + fp) else None,
    }


def clean_num(x, nd: int = 3):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def round_dict(d: dict) -> dict:
    return {k: (clean_num(v) if isinstance(v, (float, np.floating)) or v is None else v) for k, v in d.items()}


# ---------------------------------------------------------------- pipeline ---
def split_masks(idx: pd.DatetimeIndex, test_days: int, val_days: int, gap_h: int = 48) -> dict:
    end = idx.max()
    test_start = end - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    gap = pd.Timedelta(hours=gap_h)
    return {"train": idx < val_start - gap,
            "val": (idx >= val_start) & (idx < test_start - gap),
            "test": idx >= test_start,
            "trainval": idx < test_start - gap}


def run_province(key: str, cfg: dict, src, test_days: int, val_days: int) -> dict:
    hist = clean(src.history(key, cfg), f"{key}/history")
    base = make_features(hist)
    models, metrics, chosen = {}, {}, {}
    for h in HORIZONS:
        X_all = add_target_features(base, h)
        y_all = hist["pm2_5"].shift(-h)
        ok = X_all.notna().all(axis=1) & y_all.notna()
        X, y = X_all[ok], y_all[ok]
        cols = list(X.columns)
        m = split_masks(X.index, test_days, val_days)
        if m["train"].sum() < MIN_TRAIN_ROWS or m["test"].sum() < MIN_TEST_ROWS:
            raise ValueError(f"{key} h={h}: not enough data (train={m['train'].sum()}, test={m['test'].sum()})")
        tgt_t = X.index + pd.Timedelta(hours=h)

        # 1) choose the model type on the validation window (trained on train only)
        val_mae = {}
        for kind in ("ridge", "hgb"):
            mdl = fit_point(kind, X[m["train"]], y[m["train"]])
            val_mae[kind] = float(np.mean(np.abs(predict(mdl, X[m["val"]]) - y[m["val"]].values)))
        kind = min(val_mae, key=val_mae.get)

        # 2) honest test: refit on train+val, score on the untouched last days
        mt = fit_point(kind, X[m["trainval"]], y[m["trainval"]])
        qlo = fit_quantile(X[m["trainval"]], y[m["trainval"]], 0.10)
        qhi = fit_quantile(X[m["trainval"]], y[m["trainval"]], 0.90)
        Xt, yt = X[m["test"]], y[m["test"]].values
        p = predict(mt, Xt)
        lo, hi = np.minimum(predict(qlo, Xt), p), np.maximum(predict(qhi, Xt), p)
        persist = Xt["pm_lag0"].values
        clim = climatology(y[m["trainval"]], tgt_t[m["trainval"]], tgt_t[m["test"]])
        res = evaluate(yt, p, persist)
        res["interval80_coverage"] = float(np.mean((yt >= lo) & (yt <= hi)))
        res["baseline_persistence_mae"] = float(np.mean(np.abs(persist - yt)))
        res["baseline_climatology_mae"] = float(np.mean(np.abs(clim - yt)))
        res["model"], res["val_mae"] = kind, val_mae
        res["test_from"] = str(Xt.index.min())[:16]
        res["test_to"] = str(Xt.index.max())[:16]
        # 2b) hard mode: train ONLY on data before the last Feb-Apr haze season, test on it
        s0, s1 = last_season(now_local())
        tr_m = X.index < s0 - pd.Timedelta(hours=48)
        se_m = (X.index >= s0) & (X.index <= s1)
        res["season"] = None
        if tr_m.sum() >= MIN_TRAIN_ROWS and se_m.sum() >= MIN_TEST_ROWS:
            ms = fit_point(kind, X[tr_m], y[tr_m])
            Xs, ys = X[se_m], y[se_m].values
            sres = evaluate(ys, predict(ms, Xs), Xs["pm_lag0"].values)
            sres["baseline_persistence_mae"] = float(np.mean(np.abs(Xs["pm_lag0"].values - ys)))
            sres["mean_pm25"] = float(ys.mean())
            sres["from"], sres["to"] = str(s0)[:10], str(s1)[:10]
            res["season"] = round_dict(sres)
            LOG.info("%s +%2dh haze season %s..%s: MAE %.2f | persistence %.2f | skill %+.0f%% | mean PM2.5 %.1f",
                     key, h, sres["from"], sres["to"], sres["mae"], sres["baseline_persistence_mae"],
                     100 * (sres["skill_vs_persistence"] or 0), sres["mean_pm25"])
        else:
            LOG.warning("%s +%dh: not enough data for haze-season test", key, h)
        metrics[h] = res
        LOG.info("%s +%2dh [%s] test MAE %.2f | persistence %.2f | climatology %.2f | skill %+.0f%% | 80%% cover %.0f%%",
                 key, h, kind, res["mae"], res["baseline_persistence_mae"], res["baseline_climatology_mae"],
                 100 * (res["skill_vs_persistence"] or 0), 100 * res["interval80_coverage"])

        # 3) final models use ALL data for the live forecast
        models[h] = (fit_point(kind, X, y), fit_quantile(X, y, 0.10), fit_quantile(X, y, 0.90), cols)
        chosen[h] = kind

    # --- live forecast from the latest data -----------------------------------
    rec = clean(src.recent(key, cfg), f"{key}/recent")
    rec = rec[rec.index <= now_local()]
    feats = make_features(rec).dropna()
    if feats.empty:
        raise ValueError(f"{key}: no complete feature row in recent data")
    origin = feats.index[-1]
    if now_local() - origin > pd.Timedelta(hours=6):
        raise ValueError(f"{key}: latest usable data is {origin} (stale)")
    forecasts = []
    for h in HORIZONS:
        mp, ml, mh, cols = models[h]
        row = add_target_features(feats.loc[[origin]], h)[cols]
        p = float(predict(mp, row)[0])
        lo, hi = min(float(predict(ml, row)[0]), p), max(float(predict(mh, row)[0]), p)
        forecasts.append({"horizon_h": h, "target_time": str(origin + pd.Timedelta(hours=h))[:16].replace(" ", "T"),
                          "pm25": clean_num(p, 1), "low": clean_num(lo, 1), "high": clean_num(hi, 1),
                          "level": level_of(p)})
    return {"name": cfg["name"], "origin_time": str(origin)[:16].replace(" ", "T"),
            "current_pm25": clean_num(rec.loc[origin, "pm2_5"], 1), "forecasts": forecasts,
            "metrics": {str(h): round_dict({k: v for k, v in metrics[h].items() if k != "val_mae"}) for h in HORIZONS},
            "models": {str(h): chosen[h] for h in HORIZONS}}


def last_season(today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Most recent completed haze season (Feb 1 - Apr 30)."""
    year = today.year if today.month >= 5 else today.year - 1
    return pd.Timestamp(year, 2, 1), pd.Timestamp(year, 4, 30, 23)


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="forecast.json")
    ap.add_argument("--provinces", nargs="*", default=list(PROVINCES), choices=list(PROVINCES))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--val-days", type=int, default=60)
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data, offline")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    src = SyntheticSource() if a.selftest else OpenMeteoSource(a.cache_dir)
    out_path = Path(tempfile.gettempdir()) / "selftest_forecast.json" if a.selftest else Path(a.out)
    provinces = a.provinces[:1] if a.selftest else a.provinces
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "data_source": "Synthetic (selftest)" if a.selftest else "Open-Meteo Air Quality API (CAMS model data)",
              "horizons": HORIZONS, "test_days": a.test_days, "provinces": {}, "errors": {}}
    for key in provinces:
        try:
            result["provinces"][key] = run_province(key, PROVINCES[key], src, a.test_days, a.val_days)
        except Exception as exc:  # one province failing must not kill the others
            LOG.error("%s failed: %s", key, exc)
            LOG.debug("details", exc_info=True)
            result["errors"][key] = str(exc)
    if not result["provinces"]:
        LOG.error("no province succeeded; not writing %s", out_path)
        return 1
    # keep previous good results for provinces that failed this time
    if not a.selftest and out_path.exists() and result["errors"]:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            for k in result["errors"]:
                if k in old.get("provinces", {}):
                    result["provinces"][k] = old["provinces"][k]
                    result["provinces"][k]["stale"] = True
        except (OSError, ValueError):
            pass
    write_json_atomic(out_path, result)
    LOG.info("wrote %s (%d province(s), %d error(s))", out_path, len(result["provinces"]), len(result["errors"]))
    if a.selftest:
        check_selftest(out_path)
    return 0


def check_selftest(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provinces"], "no provinces"
    for p in data["provinces"].values():
        assert len(p["forecasts"]) == len(HORIZONS)
        for f in p["forecasts"]:
            assert f["low"] <= f["pm25"] <= f["high"], f
            assert f["level"] in LEVELS
        for h in HORIZONS:
            m = p["metrics"][str(h)]
            assert m["mae"] is not None and m["n"] >= MIN_TEST_ROWS, m
            assert "season" in m, m
    LOG.info("SELFTEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
