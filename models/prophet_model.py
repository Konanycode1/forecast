"""Prophet réentraîné chaque jour à minuit (section 4 du guide)."""
import logging
import time

import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetModel:
    min_history = 1

    def __init__(self, cps=0.05, sps=10.0, daily_order=10, weekly_order=3, window_days=28):
        self.kw = dict(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
        self.do, self.wo, self.win = daily_order, weekly_order, window_days
        self.cache, self.next_refit = None, None
        self.fit_times = []                          # coût à rapporter (6.4)

    def _refit(self, hist):
        t0 = time.perf_counter()
        d = hist.iloc[-self.win * 1440:].dropna()    # Prophet tolère les trous
        d = pd.DataFrame({"ds": d.index, "y": d.values})
        m = Prophet(daily_seasonality=False, weekly_seasonality=False,
                    yearly_seasonality=False, **self.kw)
        m.add_seasonality("daily", period=1, fourier_order=self.do)
        m.add_seasonality("weekly", period=7, fourier_order=self.wo)
        m.fit(d)
        now = hist.index[-1]
        fut = pd.DataFrame({"ds": pd.date_range(now, periods=24 * 60 + 16, freq="min")[1:]})
        self.cache = m.predict(fut).set_index("ds")[["yhat", "yhat_upper"]]
        self.next_refit = now.normalize() + pd.Timedelta(days=1)
        self.fit_times.append(time.perf_counter() - t0)

    def fit(self, y, y_val=None):
        self._refit(y)
        return self

    def forecast(self, hist, h):
        if hist.index[-1] >= self.next_refit:
            self._refit(hist)
        fut = pd.date_range(hist.index[-1], periods=h + 1, freq="min")[1:]
        return self.cache.loc[fut, "yhat"].values
