# Bank Marketing: Term Deposit Prediction

Predicts which clients of a Portuguese bank will subscribe to a term deposit after a phone
marketing campaign, so the call center can work down a ranked list instead of calling in
random order.

Data: [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing),
45,211 clients, 16 features plus the target `y`. A local copy lives in `data/bank-full.csv`.

## How to run

- `pip install -r requirements.txt`
- Open `bank_marketing.ipynb` and run all cells top to bottom, runtime is about
  two minutes

## Approach

- `duration` (length of the last call) is excluded as a feature: it is only known after the
  call, so using it would be leakage. The strongest raw predictor, dropped deliberately.
- 11.7% of clients subscribe, so accuracy is useless as a metric. Primary metric is PR-AUC
  (baseline 0.117 for random ranking), with recall at a fixed call budget as the secondary view.
- Two models in one shared pipeline (scaling + one-hot encoding):
  logistic regression as the baseline, hist gradient boosting as the
  performance model (`HistGradientBoostingClassifier`). Small grid search, 5-fold CV on the
  training data only.
- 60/20/20 split: decision thresholds are tuned on the validation set to maximize F1, the
  test set is scored once at the end.

## Findings

On the held-out test set of 9,043 clients, at the tuned thresholds:

| model | PR-AUC | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| logistic regression | 0.439 | 0.778 | 0.458 | 0.442 | 0.450 |
| hist gradient boosting | 0.475 | 0.804 | 0.468 | 0.540 | 0.501 |

- The boosting model's tuned threshold of 0.62 works out to calling the top 13.5% of the ranked
  list: 46.8% of those calls convert (4x the 11.7% base rate) and 54% of all would-be subscribers
  are caught with a seventh of the calling effort. The top 20% catches 62.4% at 36.5% conversion.
- Top drivers: outcome of the previous campaign, call month, known phone line, then
  demographics (retirees and students subscribe far more often).
- Recommended operating point: start with the top 13.5% cut and extend toward 20% if the call
  center has spare capacity; beyond 30% precision decays toward the base rate.

Caveats are in the notebook (management summary and follow-up questions at the end): the month
effect partly reflects the campaign's own scheduling, and a proper next step would be uplift
modeling with a holdout group that is not called.
