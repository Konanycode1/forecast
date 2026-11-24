"""LSTM à sortie directe sur 15 minutes (section 5 du guide)."""
import numpy as np
import torch
import torch.nn as nn


def calendar(index):
    m = (index.hour * 60 + index.minute).to_numpy()
    d = index.dayofweek.to_numpy() * 1440 + m
    return np.column_stack([np.sin(2 * np.pi * m / 1440), np.cos(2 * np.pi * m / 1440),
                            np.sin(2 * np.pi * d / 10080), np.cos(2 * np.pi * d / 10080)])


class Net(nn.Module):
    def __init__(self, n_in=5, hidden=64, layers=1, dropout=0.0, H=15):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])                 # dernière position -> 15 sorties


class LSTMModel:
    def __init__(self, L=120, hidden=64, layers=1, lr=1e-3, epochs=30,
                 batch=256, patience=5, seed=0, H=15):
        self.L, self.H, self.seed = L, H, seed
        self.hp = dict(hidden=hidden, layers=layers, dropout=0.2 if layers > 1 else 0.0)
        self.lr, self.epochs, self.batch, self.patience = lr, epochs, batch, patience
        self.min_history = L
        self.best_epoch = epochs

    def _features(self, y):
        z = (y.to_numpy() - self.mu) / self.sd
        return np.column_stack([z, calendar(y.index)]).astype(np.float32)

    def _windows(self, y):
        F = self._features(y)
        X, Y = [], []
        for i in range(self.L, len(F) - self.H + 1):
            x, t = F[i - self.L:i], F[i:i + self.H, 0]
            if not (np.isnan(x).any() or np.isnan(t).any()):   # pas de trou
                X.append(x)
                Y.append(t)
        return torch.tensor(np.array(X)), torch.tensor(np.array(Y))

    def fit(self, y_train, y_val=None):
        """y_val doit commencer L minutes avant la validation (contexte)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.mu, self.sd = np.nanmean(y_train), np.nanstd(y_train)
        Xtr, Ytr = self._windows(y_train)
        self.net = Net(H=self.H, **self.hp)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn, best, wait, state = nn.MSELoss(), np.inf, 0, None
        if y_val is not None:
            Xva, Yva = self._windows(y_val)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(Xtr), self.batch):
                idx = perm[b:b + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[idx]), Ytr[idx])
                loss.backward()
                opt.step()
            if y_val is None:
                continue
            self.net.eval()
            with torch.no_grad():
                v = loss_fn(self.net(Xva), Yva).item()
            print(f"  époque {ep + 1} : perte validation {v:.4f}", flush=True)
            if v < best - 1e-5:
                best, wait, self.best_epoch = v, 0, ep + 1
                state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if state is not None:
            self.net.load_state_dict(state)          # meilleure époque en validation
        return self

    def forecast(self, hist, h):
        x = torch.tensor(self._features(hist.iloc[-self.L:])).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            z = self.net(x).numpy()[0][:h]
        return z * self.sd + self.mu                 # retour à l'échelle log
