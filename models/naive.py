"""Références naïves (section 3.1 du guide)."""


class Persistence:
    """y(t+h) = y(t) : ce que fait en pratique le HPA réactif."""
    min_history = 1

    def fit(self, y, y_val=None):
        return self

    def forecast(self, hist, h):
        return [hist.iloc[-1]] * h


class SeasonalNaive:
    """y(t+h) = y(t+h-1440) : la même minute la veille."""
    min_history = 1440

    def __init__(self, m=1440):
        self.m = m

    def fit(self, y, y_val=None):
        return self

    def forecast(self, hist, h):
        return list(hist.iloc[-self.m:-self.m + h].values)
