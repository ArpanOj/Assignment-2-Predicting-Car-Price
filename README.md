# A2: Predicting Car Price — From-Scratch Regression, Experiment Tracking, and Deployment

Submission for **A2: Predicting Car Price** (AT82.03: Machine Learning). Builds on A1's
car price prediction system with a linear regression model implemented from scratch
(gradient descent, no sklearn model fitting), a 144-configuration hyperparameter sweep
tracked with MLflow, and a second, independently deployable web page serving the new
model alongside the original.

**Live deployment:** https://web-st127304.ml.brain.cs.ait.ac.th/ (original model at `/`,
new model at `/v2`)

> This is a separate repository from my A1 submission — no A1 code is duplicated here
> except the data cleaning/preprocessing pipeline, reused per the assignment's
> instructions ("replace the modeling part with the class we have built above").

---

## Repository structure

```
.
├── README.md
├── Task_1_Implementation.ipynb    # Task 1: modified LinearRegression class
├── Task_2_Experimentation.ipynb   # Task 2: MLflow sweep, report, feature importance
├── A2-Predicting Car Price.pdf    # assignment brief
├── Cars.csv                       # dataset
└── App/
    ├── Dockerfile
    ├── docker-compose.yaml        # local development config
    └── Code/
        ├── app.py                    # multi-page Dash app (original model + new model)
        ├── car_model.py              # A1's sklearn model wrapper (unchanged)
        ├── car_model_v2.py           # Task 2 model wrapper (handles polynomial features)
        ├── regression_models.py      # from-scratch LinearRegression/Lasso/Ridge/NoRegularization
        ├── Car_Price_Model.pkl       # A1's trained SVR model
        └── Car_Price_Model_V2.pkl    # Task 2's trained best model
```

---

## Task 1 — Custom `LinearRegression` class

Building on the course's `03 - Regularization.ipynb` template, `LinearRegression` was
extended with:

- **`r2()`** — R² scoring, alongside the existing MSE.
- **`weight_initialise()`** — `'zeros'` (original default) or `'xavier'` (Glorot-style
  uniform initialization, `U[-1/√n, 1/√n]`), selectable via `initialise_method`.
- **Momentum** — optional, configurable momentum term in the gradient update
  (`θ ← θ - lr·∇J + momentum·v_prev`), toggled via `use_momentum`/`momentum`.
- **`plot_feature_importance()`** — horizontal bar chart of `|coefficient|` per feature.
- **`NoRegularization`** — a zero-penalty regularizer class, added so plain
  (unregularized) linear regression shares the same interface as `Lasso`/`Ridge`; required
  for Task 2's "normal" comparison condition, since `_train()` always calls
  `self.regularization.derivation(theta)` and needs *some* object to call it on.
- **Bias exclusion from regularization** — the gradient computation excludes `theta[0]`
  (the bias/intercept) from every regularizer's penalty. This matters because the model's
  target, `log(selling_price)`, requires a large intercept (~13) to fit at all; penalizing
  it would actively fight the model's ability to reach the correct prediction scale.

See `Task_1_Implementation.ipynb` for the full class and inline documentation.

## Task 2 — Experiment with MLflow

The A1 preprocessing pipeline (cleaning, imputation, one-hot encoding, standardization,
log-transforming the target) was reused as the starting point, then adapted to feed the
custom `LinearRegression` class — numpy arrays with an explicit intercept column, rather
than the DataFrame input sklearn's models accept.

**Comparison performed** (144 total configurations, tracked via MLflow):
- **Model type**: `normal`, `lasso`, `ridge`, `polynomial` (degree-2 expansion on the 7
  numeric features, rescaled post-expansion to prevent gradient overflow)
- **Batch method**: `batch`, `mini`-batch, `sto`chastic
- **Weight initialization**: `zeros`, `xavier`
- **Momentum**: on / off
- **Learning rate**: `0.01`, `0.001`, `0.0001`

Each configuration was evaluated via 5-fold cross-validation (MSE and R², both logged to
MLflow per fold and averaged).

**Best configuration found:** `polynomial` features, mini-batch gradient descent, Xavier
initialization, no momentum, `lr=0.01`.
- Cross-validated: MSE = 0.0677 (log-price space), R² = 0.884
- Held-out test set: MSE ≈ 4.06×10¹⁰ (rupee² space), R² = 0.854, RMSE ≈ ₹201,600

See `Task_2_Experimentation.ipynb` for the full sweep, MLflow screenshots, final
comparison table, and discussion of findings (including the CV-vs-test performance gap
and the numerical stability challenges of polynomial features under stochastic gradient
descent).

## Task 3 — Deployment

The web application (Dash) serves **two** pages from one app:
- `/` — the original A1 model (tuned SVR via sklearn)
- `/v2` — the new Task 2 model (from-scratch linear regression, best config per the MLflow
  sweep above)

Both pages share the same input form structure and the same "allow blank fields, impute
with training-set median/mode" behavior as A1.

**Local run:**
```bash
cd App
docker compose up --build
```
Then visit `http://localhost:8050/` and `http://localhost:8050/v2`.

**Live deployment:** hosted behind a Traefik reverse proxy on the AIT CSIM `ml-brain`
server. The compose file used on the server includes Traefik routing labels for the
`web-st127304.ml.brain.cs.ait.ac.th` subdomain (not identical to the local
`docker-compose.yaml` above, which is for local development only — the server-side
config additionally routes through Traefik for SSL termination and subdomain-based
request routing, per the course's shared infrastructure setup).

---

## Design notes

**Why the model classes live in `regression_models.py` instead of the notebook.**
Python's `pickle` module saves a reference to a class's module and name, not its actual
code. Classes defined inline in a notebook are only importable from that notebook's own
runtime session — a pickled model built from them can't be unpickled anywhere else,
including in the deployed app. Moving `LinearRegression`/`Lasso`/`Ridge`/
`NoRegularization` into a real, importable `.py` file — used by both the notebook (for
training) and the app (for inference) — resolves this. This mirrors why A1's
`CarPriceModel` needed its own `car_model.py` rather than living in the notebook.

**Why `car_model_v2.py`'s one-hot encoding uses `drop_first=False` at inference despite
training with `drop_first=True`.** One-hot encoding a single new row with
`drop_first=True` would treat whichever category happens to be present as "the first
one" and drop it — silently zeroing out that field instead of encoding it correctly.
Using `drop_first=False` at inference, then `reindex()`-ing against the training column
list (which *was* built with `drop_first=True`), correctly reproduces the training
feature space without this trap. A1's `car_model.py` solves the identical problem the same
way.

**Why polynomial-feature inference needs the `poly`/`poly_scaler` objects stored inside
the model wrapper.** Since the winning model type was `polynomial`, `CarPriceModelV2`
needs to re-apply the same degree-2 expansion and post-expansion rescaling to new input
at prediction time — using the exact fitted transformers from training (never re-fit at
inference), consistent with the "fit on train only" discipline used throughout the
pipeline.

---

## Deliverables checklist

- [x] Jupyter notebook(s): Task 1 class modifications; Task 2 experiment, report, and
      MLflow screenshots
- [x] This `README.md`
- [x] `App/` folder with `Dockerfile`, `docker-compose.yaml`, and `Code/` containing both
      models and the multi-page app
- [x] Live deployment on `ml.brain.cs.ait.ac.th`
https://web-st127304.ml.brain.cs.ait.ac.th
