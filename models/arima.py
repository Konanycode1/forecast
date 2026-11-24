"""ARIMA avec saisonnalités par termes de Fourier (section 3.2 du guide)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fourier(index, period, K):
    t = ((index - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)).to_numpy()
    cols = []
    for k in range(1, K + 1):
        w = 2 * np.pi * k * t / period
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols) if cols else np.empty((len(index), 0))


class FourierARIMA:
    min_history = 1

    def __init__(self, order=(2, 0, 1), K_day=5, K_week=3, fit_days=21):
        self.order, self.Kd, self.Kw, self.fit_days = order, K_day, K_week, fit_days

    def X(self, index):
        return np.hstack([fourier(index, 1440, self.Kd), fourier(index, 10080, self.Kw)])

    def fit(self, y, y_val=None):
        y = y.iloc[-self.fit_days * 1440:]          # fenêtre récente
        model = SARIMAX(y.values, exog=self.X(y.index), order=self.order,
                        trend="c", enforce_stationarity=False)
        self.res = model.fit(disp=False, maxiter=200)
        self.last_t = y.index[-1]
        return self

    def forecast(self, hist, h):
        new = hist.loc[self.last_t:].iloc[1:]       # observations arrivées depuis
        if len(new):
            # met à jour l'état du filtre de Kalman sans réestimer les paramètres
            self.res = self.res.extend(new.values, exog=self.X(new.index))
            self.last_t = new.index[-1]
        fut = pd.date_range(self.last_t, periods=h + 1, freq="min")[1:]
        return np.asarray(self.res.forecast(steps=h, exog=self.X(fut)))
