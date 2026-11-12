# Predicting Airline Satisfaction — DAM2026 exam project

**Competition:** [Kaggle Playground Series S6E10 — Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)
*(topic approved by the course instructor)*

## Task
Each row is one airline passenger: their profile (age, gender, loyal or not), their trip
(business or personal travel, class, distance, delays) and the 0–5 ratings they gave to
13 services (wifi, seat comfort, online boarding, cleanliness…).
We predict the **probability that the passenger is satisfied**.

* **Target:** `satisfaction` — True (satisfied) / False (neutral or dissatisfied). 44.4% True.
* **Metric:** **ROC AUC** on the predicted probabilities.
  In plain words: take one satisfied and one unsatisfied passenger at random — AUC is the
  chance the model ranks the satisfied one higher. 0.5 = guessing, 1.0 = perfect. It rewards
  correct *ranking*, not a 0.5 cut-off, and is not affected by class imbalance.
* **Data:** 699,635 training rows, 299,844 test rows, 22 features (synthetic data generated
  by Kaggle from the original *Airline Passenger Satisfaction* survey).

## Team and roles
| Member | Role | Owns |
|---|---|---|
| Harisson Tenu | Data & EDA | `notebooks/01_data_eda.ipynb`, figures, data description |
| Emmanuel Gyamfi Kwarteng | Preprocessing & validation | `src/preprocess.py`, `notebooks/02_preprocessing_validation.ipynb` |
| Eno Daniel Simeon | Modelling & submissions | `notebooks/03_baselines_submission.ipynb`, Kaggle submissions |

## Repository structure
```
data/            competition CSVs (not committed — download from Kaggle)
src/preprocess.py  shared cleaning, encoding and scaling
notebooks/
  01_data_eda.ipynb                 data description, EDA, missing values & anomalies
  02_preprocessing_validation.ipynb encoding, scaling, 5-fold CV scheme
  03_baselines_submission.ipynb     baselines, logistic regression, LightGBM, submission
figures/         plots and the results table used in the report
submissions/     files uploaded to Kaggle
report/          intermediate status report slides
```

## How to run
```bash
pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e10 -p data && unzip data/*.zip -d data
# run notebooks 01 → 02 → 03 from the notebooks/ folder
```

## Current results (5-fold stratified CV, ROC AUC)
| Model | OOF ROC AUC | Fold std |
|:--|--:|--:|
| Constant (class prior) — trivial baseline | 0.5000 | 0.0000 |
| Rule: Online boarding rating as score | 0.8404 | 0.0011 |
| Rule: Business class + Online boarding | 0.8802 | 0.0011 |
| Logistic regression (scaled, one-hot) | 0.9368 | 0.0005 |
| LightGBM (untuned, early stopping) | **0.9587** | 0.0007 |

Kaggle public leaderboard (LightGBM baseline): **`FILL IN AFTER UPLOAD`**

## References
* Original dataset: [Airline Passenger Satisfaction (Kaggle)](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
* No public notebooks were reused so far. Any code we base on public work will be credited here.
