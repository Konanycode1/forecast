"""Évaluation glissante et métriques (sections 2.3 et 6 du guide)."""
import time

import numpy as np
import pandas as pd

H = 15      # horizon en pas d'une minute


def rolling_eval(model, y_log, start, end, step=15, h=H):
    """y_log : série complète en log1p. Une origine toutes les `step` minutes
    entre start et end ; à chaque origine le modèle ne voit que le passé."""
    print("test en cours ....")
    rows, lat = [], []
    sub = y_log.loc[start:end]
    origins = sub.index[:-h:step]
    for k, t in enumerate(origins):
        hist = y_log.loc[:t]
        if hist.iloc[-model.min_history:].isna().any():
            continue                                 # origine dans un trou
        t0 = time.perf_counter()
        pred_log = model.forecast(hist, h)
        lat.append(time.perf_counter() - t0)
        future = y_log.loc[t:].iloc[1:h + 1]
        for hh, (ts, yp) in enumerate(zip(future.index, pred_log), start=1):
            yv = future.loc[ts]
            if np.isnan(yv):
                continue
            rows.append({"origin": t, "target_time": ts, "h": hh,
                         "y": float(np.expm1(yv)), "y_hat": max(float(np.expm1(yp)), 0.0)})
        if k % 200 == 0:
            print(f"  origine {k + 1}/{len(origins)}", flush=True)
    return pd.DataFrame(rows), np.array(lat)


def metrics(df, c_pod, h=H, mape_floor=1.0):
    d = df[df.h == h].sort_values("target_time")
    y, yh = d.y.to_numpy(), d.y_hat.to_numpy()
    e = y - yh
    keep = y >= mape_floor
    peak = y >= np.quantile(y, 0.9)
    r = lambda v: np.ceil(v / c_pod)
    return {
        "n": int(len(y)),
        "biais": float(np.mean(e)),                  # > 0 : sous-estimation moyenne
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAPE": float(100 * np.mean(np.abs(e[keep] / y[keep]))),
        "MAPE_exclus": int((~keep).sum()),
        "WAPE": float(100 * np.abs(e).sum() / y.sum()),
        "MAE_pic": float(np.mean(np.abs(e[peak]))),
        "sous_estim_pic_%": float(100 * np.mean(yh[peak] < y[peak])),
        "deficit_replicas_pic": float(np.mean(np.maximum(0, r(y[peak]) - r(yh[peak])))),
    }


def mae_by_horizon(df):
    return df.assign(ae=(df.y - df.y_hat).abs()).groupby("h").ae.mean()
