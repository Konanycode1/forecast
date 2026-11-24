"""Export du débit et du CPU observés depuis Prometheus (section 1.3).

Usage : python export_prometheus.py RUN_ID 2026-11-20T10:00:00Z 2026-11-20T16:00:00Z
"""
import sys

import pandas as pd
import requests

PROM = "http://prometheus.monitoring:9090"   # [À COMPLÉTER]
QUERIES = {
    "rps": 'sum(rate(http_requests_total{namespace="bench"}[1m]))',
    "cpu_cores": 'sum(rate(container_cpu_usage_seconds_total{namespace="bench", container="api"}[1m]))',
}


def query_range(q, start, end, step="15s"):
    r = requests.get(f"{PROM}/api/v1/query_range",
                     params={"query": q, "start": start, "end": end, "step": step}, timeout=60)
    r.raise_for_status()
    res = r.json()["data"]["result"]
    if not res:
        return pd.Series(dtype=float)
    ts, vals = zip(*res[0]["values"])
    return pd.Series([float(v) for v in vals],
                     index=pd.to_datetime([float(t) for t in ts], unit="s"))


if __name__ == "__main__":
    run_id, start, end = sys.argv[1:4]
    df = pd.DataFrame({k: query_range(q, start, end) for k, q in QUERIES.items()})
    df.to_csv(f"data/prometheus/{run_id}.csv", index_label="timestamp")
    print(df.resample("1min").mean().describe())
