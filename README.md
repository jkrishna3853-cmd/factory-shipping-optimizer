# Factory Reallocation & Shipping Optimization Recommendation System

A decision-support system for the Nassau Candy Distributor case: predict shipping lead time, simulate moving a product to another factory, and recommend a reassignment only when the predicted gain is distinguishable from model noise.

**Headline finding:** in this dataset, lead time is driven almost entirely by **ship mode**. Factory-to-customer distance has no measurable effect, so the system recommends **no** reassignments. That result is reported as found, not forced into a success story.

## Business problem

Products are assigned to factories by static rules. The goal is to find out whether reassigning products to other factories would cut lead time without eroding profit.

## Dataset

- `Nassau Candy Distributor.csv`: 10,194 order rows, 8,549 orders, 15 products, 5 factories, 4 customer regions, 4 ship modes, 2 countries (US, Canada).
- Factory coordinates and the product-to-factory mapping come from the project brief (`data/external/factories.csv`, `data/external/product_factory.csv`).
- Customer coordinates are postal-code centroids from GeoNames (CC BY), matched to every order (0 unmatched). US ZIPs lost their leading zeros upstream (449 rows) and are restored by padding to 5 digits.

## Data findings that shaped the design

1. **Ship dates are shifted by whole years.** Raw lead time is 904 to 1,642 days, in three bands one year apart. We remove the whole-year shift (an explicit ASSUMPTION) to get an adjusted lead time of 173 to 185 days (mean 177.6). Raw dates are kept in the data.
2. **Each product is made at exactly one factory**, so factory and product carry the same information. A model cannot learn a factory's own effect. Factory is not a model input; factory-to-customer **distance** is used instead.
3. **The data is lopsided:** two factories hold 97% of the rows.
4. **Gross Profit = Sales - Cost on every row.** There is no shipping cost or factory-specific production cost, so profit impact is **not** estimated unless the user supplies an assumed cost per unit-km.

## Methodology

| Stage | What was done |
|---|---|
| Features | Units, Sales, Cost, distance_km, order month, order day-of-week, Ship Mode, Region, Product Name |
| Leakage control | Ship Date, raw lead time and shift columns are never inputs; the train/test split is grouped by Order ID |
| Models | Linear Regression, Random Forest, Gradient Boosting |
| Selection | 5-fold grouped cross-validation on training data; the test set is used once at the end |
| Outliers | Removing them (3.0 and 1.5 x IQR) did not change test error, so all rows are kept |

## Results

Test set (2,062 rows), adjusted lead time (target standard deviation 1.79 days):

| Model | Test MAE (days) | Test RMSE | Test R2 |
|---|---|---|---|
| Predict the mean | 1.433 | 1.793 | -0.00 |
| Linear Regression (ship mode only) | 0.907 | 1.091 | 0.63 |
| Linear Regression (9 features) | 0.903 | 1.091 | 0.63 |
| Random Forest | 0.871 | 1.082 | 0.636 |
| **Gradient Boosting (final)** | **0.890** | **1.085** | **0.634** |

Cross-validated MAE: Linear 0.941, Random Forest 0.914 and 0.924, Gradient Boosting 0.926 (fold-to-fold spread about 0.023), so the tree models are tied. Gradient Boosting was chosen for matching accuracy with the smallest train/test gap (Random Forest memorizes: train MAE 0.628 vs test 0.871).

**What drives lead time (permutation importance, days of extra error):** Ship Mode 0.912; every other feature is 0.010 or less. Distance contributes 0.006.

**Distance vs lead time:** Pearson 0.0032, Spearman 0.0014; mean adjusted lead time is flat across five distance groups.

**Clustering:** K-means on per-route excess lead time (beyond what ship mode predicts) gave silhouette 0.542, which does **not** beat shuffled data (mean 0.593). Per-route z-tests found 1 of 25 routes beyond |z| > 2 (1.1 expected by chance) and none beyond 3.1. There are no consistently slow routes beyond ship mode.

**Recommendation engine and KPIs** (priority = speed, 15 products):

| KPI | Value |
|---|---|
| Recommendation coverage | 0% (0 of 15 products) |
| Lead time reduction, best available but NOT confident | 0.0062% |
| Scenario confidence (order-weighted) | 0.22 |
| Profit impact stability | 85% (over ASSUMED cost levels only) |

The best predicted saving for any product was under 0.13 days (and under 0.02 days for the five large Wonka Bar products), well inside the model's typical error.

## Recommendation logic

score = w_speed x speed + w_risk x risk + w_profit x profit, with weights summing to 1. `w_risk` = 0.2; the remaining 0.8 is split by the Speed-to-Profit slider.

- speed = days saved / 0.89 (the model MAE), clipped to [-1, 1]
- risk = -(1 - share of new distances inside the range the product has shipped over)
- profit = assumed profit impact / observed gross profit, clipped to [-1, 1]

A move is recommended only if it saves at least the minimum worthwhile days (placeholder 0.25, a business decision), has confidence of at least 0.5, and scores above 0. No optimization solver is used: with 5 factories and independent product decisions, exhaustive ranking is exact, and an LP or MIP would add nothing.

## Dashboard (Streamlit)

Modules: Factory Optimization Simulator, What-If Scenario Analysis, Recommendation Dashboard (KPIs, ranked table, profit sensitivity), Risk and Impact Panel, and sidebar optimization controls (product, region, ship mode, speed-profit priority, assumed shipping cost).

Screenshots: add images to `docs/screenshots/` and link them here.

## Install and run

Download `US.txt` and `CA.txt` from https://download.geonames.org/export/zip/ into `data/external/geonames/`, put the dataset in `data/raw/`, then run these in order:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m scripts.build_processed
    python -m scripts.build_geo_dataset
    python -m scripts.make_split
    python -m scripts.train_final
    streamlit run app\streamlit_app.py
    python -m pytest -v

## Project structure

    src/        data_processing, geo, features, train, evaluation, clustering,
                simulation, recommendations, kpis
    scripts/    reproducible pipeline and analysis scripts
    app/        streamlit_app.py
    tests/      test_core.py (9 tests)
    data/       raw, external, processed (processed is not committed)

## Limitations

- Ship dates appear shifted by whole years; the correction is an assumption.
- Distance is straight-line, from postal-area centroids, not road distance.
- Each product has one factory in the data, so the effect of a different factory is extrapolated from distance alone. This cannot be validated.
- There is no shipping or production cost data; profit sensitivity uses illustrative costs.
- Three factories have fewer than 220 order rows, so their estimates are weak.
- Scenario confidence is an upper bound: it assumes independent errors.

## Future improvements

- Real shipping-cost and factory-capacity data for genuine profit optimization.
- Carrier or route-level data, or data where products ship from several factories.
- Road-distance and transit-time estimates instead of straight-line distance.
- SHAP explanations and an API layer if the system is deployed.

Postal-code coordinates: GeoNames (https://www.geonames.org), CC BY.