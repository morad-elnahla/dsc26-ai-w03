# Results: Tawseel failed-delivery baseline

## Baseline table

Last 20% of parcels by time (800 parcels), `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|
| Dummy (prior) | 0.500 | 0.220 | 0.172 | 0.22 |
| Rule: vague address text -> 0.6 risk, else 0.1 | 0.548 | 0.241 | 0.205 | 0.22 |
| My model (logistic regression + history) | 0.631 | 0.322 | 0.166 | 0.22 |

## Three sentences for Sherif

On the most recent 20% of parcels, the model finds failing deliveries clearly better than guessing (PR-AUC 0.32 vs 0.22), but it is a modest model, not a 99% one. We trained it only on older parcels and tested it only on newer ones, using only information known when the parcel is created, so these numbers are what you should expect in production. It cannot yet tell exactly which parcel will fail, so use it to rank parcels by risk (for example, to call the customer first), not to decide automatically.