# forecast/ : entraînement et évaluation des modèles de prévision

Code du guide « Entraînement et évaluation : ARIMA, Prophet, LSTM », prêt à lancer sur les données synthétiques
`data/series/charge_billetterie_1min.csv` (8 semaines, colonne `split` = train / validation / test).

## Installation (une fois)
```
cd forecast
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer
Toujours depuis le dossier `forecast/` :
```
python run.py --model persistence              # référence, MAE ≈ 2,09 req/s
python run.py --model seasonal
python run.py --model arima --quick            # essai rapide sur 2 jours
python run.py --model arima                    # évaluation complète sur les 14 jours de test
python run.py --model prophet
python run.py --model lstm --seed 0            # répéter avec --seed 1 à 4
python run.py --model arima --phase validation # pour régler les paramètres
```
Options : `--c-pod` (capacité d'un pod en req/s, 25 par défaut en attendant ta mesure 5.1.3),
`--step` (minutes entre deux origines, 15 par défaut ; 1 pour mesurer le retard),
`--data` (autre série, par exemple `data/series/s1.csv` produite par `prepare.py`).

## Résultats (dossier results/)
- `forecasts_<modèle>_<phase>.csv` : chaque prévision (origine, instant cible, horizon h, réel, prévu)
- `metrics_<modèle>_<phase>.json` : MAE, RMSE, MAPE, WAPE, biais, indicateurs en pic, temps et mémoire
- `mae_horizon_<modèle>_<phase>.csv` : MAE pour h = 1 à 15

## Fichiers
- `models/` : naive.py, arima.py, prophet_model.py, lstm.py
- `evaluate.py` : évaluation glissante et métriques
- `run.py` : entraîne un modèle puis l'évalue
- `prepare.py` : traces brutes (NASA-HTTP…) vers série par minute
- `export_prometheus.py` : export du débit et du CPU depuis Prometheus
