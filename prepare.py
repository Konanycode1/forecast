"""Traces brutes (Common Log Format, ex. NASA-HTTP) -> série par minute (section 1.1).

Usage : python prepare.py data/raw/NASA_access_log_Jul95 data/raw/NASA_access_log_Aug95 --k 1.0 --out data/series/s1.csv
"""
import argparse
import re

import numpy as np
import pandas as pd

TS = re.compile(r"\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}) ([-+]\d{4})\]")
NASA_GAPS = [("1995-08-01 14:52", "1995-08-03 04:36")]  # [À VÉRIFIER]


def counts_per_minute(paths):
    stamps = []
    for path in paths:
        with open(path, encoding="latin-1") as f:
            for line in f:
                m = TS.search(line)
                if m:
                    stamps.append(m.group(1))
    t = pd.to_datetime(pd.Series(stamps), format="%d/%b/%Y:%H:%M:%S")
    n = t.dt.floor("min").value_counts().sort_index()
    full = pd.date_range(n.index.min(), n.index.max(), freq="min")
    return n.reindex(full, fill_value=0).rename("n")   # minute sans requête = 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--k", type=float, default=1.0, help="facteur d'échelle (5.4.3)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = counts_per_minute(a.paths).astype(float)
    for s, e in NASA_GAPS:
        n.loc[s:e] = np.nan                          # trou exclu, jamais remplacé par 0
    rps = (a.k * n / 60).rename("rps")
    rps.to_frame().to_csv(a.out, index_label="timestamp")
    print(rps.describe())
