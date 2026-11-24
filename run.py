"""Entraîne un modèle et l'évalue par origine glissante.

Exemples (depuis le dossier forecast/) :
  python run.py --model persistence
  python run.py --model arima --phase validation      # réglage
  python run.py --model lstm --quick                  # essai rapide (2 jours de test)
Modèles : persistence, seasonal, arima, prophet, lstm
"""
import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

from evaluate import H, mae_by_horizon, metrics, rolling_eval

p = argparse.ArgumentParser()
p.add_argument("--model", required=True,
               choices=["persistence", "seasonal", "arima", "prophet", "lstm"])
p.add_argument("--data", default="data/series/charge_billetterie_1min.csv")
p.add_argument("--phase", choices=["validation", "test"], default="test",
               help="validation : entraîne sur train, évalue sur validation ; "
                    "test : entraîne sur train+validation, évalue sur test")
p.add_argument("--step", type=int, default=15, help="minutes entre deux origines")
p.add_argument("--c-pod", type=float, default=25.0, help="capacité d'un pod en req/s (5.1.3)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="évalue seulement 2 jours (mise au point)")
a = p.parse_args()

df = pd.read_csv(a.data, parse_dates=["timestamp"]).set_index("timestamp").asfreq("min")
y_log = np.log1p(df["rps"])                          # transformation commune (1.2)
tr = y_log[df.split == "train"]
val = y_log[df.split == "validation"]
test = y_log[df.split == "test"]

if a.phase == "validation":
    fit_y, ev = tr, val
else:
    fit_y, ev = pd.concat([tr, val]), test
start, end = ev.index[0], ev.index[-1]
if a.quick:
    end = start + pd.Timedelta(days=2)


def build(name):
    if name == "persistence":
        from models.naive import Persistence
        return Persistence()
    if name == "seasonal":
        from models.naive import SeasonalNaive
        return SeasonalNaive()
    if name == "arima":
        from models.arima import FourierARIMA
        return FourierARIMA()
    if name == "prophet":
        from models.prophet_model import ProphetModel
        return ProphetModel()
    from models.lstm import LSTMModel
    return LSTMModel(seed=a.seed, epochs=5 if a.quick else 30)


if a.model == "lstm":
    import torch
    torch.set_num_threads(1)                         # temps mesurés sur un cœur

model = build(a.model)
print(f"Entraînement {a.model} ({a.phase}) sur {fit_y.index[0]} -> {fit_y.index[-1]}", flush=True)
t0, c0 = time.perf_counter(), time.process_time()
if a.model == "lstm":
    # arrêt précoce sur la validation (avec L minutes de contexte avant)
    ctx = y_log.loc[val.index[0] - pd.Timedelta(minutes=model.L):val.index[-1]]
    model.fit(tr, ctx)
    if a.phase == "test" and not a.quick:
        best_epoch = model.best_epoch
        print(f"Réentraînement sur train+validation, {best_epoch} époques", flush=True)
        model = build("lstm")
        model.epochs = best_epoch
        model.fit(fit_y)
else:
    model.fit(fit_y)
fit_s, fit_cpu_s = time.perf_counter() - t0, time.process_time() - c0

print(f"Évaluation glissante {start} -> {end}, une origine toutes les {a.step} min", flush=True)
res, lat = rolling_eval(model, y_log, start, end, step=a.step)

tag = f"{a.model}_{a.phase}" + (f"_seed{a.seed}" if a.model == "lstm" else "") + ("_quick" if a.quick else "")
res.to_csv(f"results/forecasts_{tag}.csv", index=False)
out = metrics(res, c_pod=a.c_pod)
out.update({
    "fit_s": fit_s, "fit_cpu_s": fit_cpu_s,
    "infer_ms_p50": float(1e3 * np.median(lat)), "infer_ms_p95": float(1e3 * np.quantile(lat, 0.95)),
    "ram_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # Linux : Ko
})
if a.model == "prophet":
    out["prophet_refit_s_moyen"] = float(np.mean(model.fit_times))
with open(f"results/metrics_{tag}.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
mae_by_horizon(res).to_csv(f"results/mae_horizon_{tag}.csv")
print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()},
                 indent=2, ensure_ascii=False))
