# AI Usage

## Tools I used, and for what

I used Claude mainly to help me understand the task. It explained concepts like data leakage, point-in-time history features, and time splits in simple Arabic. This helped me understand the logic and figure out how to approach the solution. It also helped with an initial draft of `tawseel/model.py`, which I reviewed and tested myself.

## AI-suggested code I rejected or fixed

While reviewing the code, I noticed two important cases:

- In `add_history_features`, parcels with the same timestamp shouldn't count each other as previous history. I handled this by grouping them by customer and timestamp.
- In `time_split`, rows with the same boundary timestamp should stay together in the test set, so I used `< boundary` and `>= boundary`.

I checked both cases using the relevant tests.

## How I checked that my model has no leak

I made sure `courier_notes`, `delivered_at`, `failed`, and `parcel_id` were excluded from `FEATURE_COLUMNS` because they could cause leakage.

I also ran `uv run pytest`, and all 5 tests passed. The ROC AUC on the future 20% was around 0.63, which is much more realistic than a suspiciously high score like 0.99.