# Flood Prediction in Bangladesh — Data Science Capstone

**Daily flood-label prediction using environmental observations from the preceding three days**

> **Project status:** exploratory EDA and initial classification experiments. This is an educational data-science project, **not** an operational flood-warning service.

## 1. Project question and objective

**Can measurements from the previous three days help predict whether a flood will be reported at SW90 (Bahadurabad) during the following 24 hours?**

The project explores publicly available Bangladesh flood-related data, performs exploratory data analysis (EDA), and evaluates two introductory classification methods—**Logistic Regression** and **Random Forest**—using a **three-day sliding window**. I first tested local measurements from SW90 and then added measurements from SW17 (Chilmari) to check whether a second station could improve prediction. All models were compared with a simple seasonal baseline: **predict a flood in June, July, and August; predict no flood in other months**.

A useful model should be compared with a baseline and assessed by more than accuracy, especially because labeled flood days are rare.

## 2. Data source and provenance

The starting point is the publicly available [Bangladesh Flood Forecasting Dataset by Srabon-bot](https://github.com/Srabon-bot/Bangladesh-flood-data-set). The raw daily multistation file examined in EDA is [`data/features/2026-08-07c/all_stations.parquet`](https://github.com/Srabon-bot/Bangladesh-flood-data-set/tree/main/data/features/2026-08-07c). My EDA notebook reads the Parquet dataset, investigates target availability and environmental features, and constructs a smaller cleaned CSV for modeling.

**According to the source project's documentation**, data originate from a combination of services, not a single physical weather station:

| Provider / dataset | Role in the upstream project | Practical interpretation |
|---|---|---|
| Open-Meteo with ERA5 / ERA5-Land | Local and upstream/catchment rainfall and soil moisture | Reanalysis-based environmental estimates, not necessarily in-situ gauges |
| Open-Meteo Flood API / GloFAS reanalysis | Daily river discharge | Modeled/estimated **flow in m³/s**, **not** the observed river stage/water level |
| GFMS | Satellite/model-based indications of flooding | One flood-label evidence source |
| NASA MCDWD | MODIS/VIIRS flood detections | Additional positive flood-evidence source |
| Copernicus Global Flood Monitoring (GFM) | Sentinel-1 SAR flood detection | Additional flood-evidence source |
| Dartmouth Flood Observatory (DFO) | Historical flood event records | Historical flood information |
| Global Flood Database (GFD) | Historical mapped flood extent | Additional flood evidence |
| FFWC/BWDB annual flood reports | Official historical context / verification | Not the same as complete daily ground truth |
| Copernicus DEM GLO-30 | Terrain elevation | Source for static terrain measures including HAND-style derived topography |

The source project describes combining five flood-detection evidence sources using a **positive OR rule** (a positive indication from at least one source can create a positive flood label). Consequently, labels may vary in reliability or local coverage. A dataset `0` and a model prediction of `0` are not independent proof that flooding was absent.

Source documentation: [data description and build pipeline](https://github.com/Srabon-bot/Bangladesh-flood-data-set#how-this-dataset-was-built).

## 3. Datasets used in this project

| Version | Rows | Columns | Period | Scope |
|---|---:|---:|---|---|
| Original `all_stations.parquet` (EDA) | **839,340** | **49** | 1950–7 Aug 2026 | 30 stations; many early variables and labels missing |
| `Data_Bangaldesh_Clean.csv` (modeling) | **54,780** | **45** | 1 Jan 2021–31 Dec 2025 | 30 stations, one row per station per day, observed target subset |
| SW90 filtered from clean file | **1,826** | **45** | 2021–2025 | Bahadurabad; **53** positive 24h labels |
| SW17 filtered from clean file | **1,826** | **45** | 2021–2025 | Chilmari; **30** positive 24h labels |

In the cleaned file, all 54,780 records have `flood_within_24h_label_regime = observed`. This is a label-regime field, **not** a quality guarantee. The cleaned dataset retains `upstream_chain_discharge_lag1d`, which is missing for about **90%** of rows and was not used in the principal 3-day-window models. The cleaned CSV has no duplicate `(station_id, date)` pairs.

The EDA considered multiple basins and stations. The modeling experiments focus on **SW90** as the target location and **SW17** as an auxiliary measurement location in the Brahmaputra–Jamuna system. Their discharge patterns are closely related; however, exact hydrological travel time and independent monitoring value are not established by correlation alone.

## 4. Data dictionary — every column in `Data_Bangaldesh_Clean.csv`

**How to read this table:** names below are copied exactly from the project's 45-column cleaned CSV. A 'lag' is historical relative to the date of that row; rolling aggregates and label horizons have different meanings. For source-engineered variables where the source README does not specify an exact mathematical formula, the explanation describes the documented role without inventing a formula.

### Identity & geography

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `station_id` | Unique monitoring-station identifier (e.g., SW90 or SW17). | text |
| `date` | Calendar date of the daily station record; used to order the time series. | YYYY-MM-DD |
| `station_name` | Human-readable monitoring-station name. | text |
| `year` | Calendar year derived from `date` during EDA/cleaning. *Derived in project* | year |
| `basin` | River-basin grouping (`ganges`, `brahmaputra`, `meghna`, `cht`). *Added in project* | category |

### Daily environmental variables

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `rainfall_local_mm` | Daily rainfall estimate at/near the station, from weather reanalysis; not necessarily a ground rain-gauge measurement. | mm |
| `rainfall_upstream_mm` | Daily estimate of rainfall across upstream/catchment locations contributing to the station; spatially aggregated. | mm |
| `soil_moisture_local` | Local modeled/reanalysis soil-moisture value; higher generally means wetter soil. The precise soil layer and unit should be checked in the source pipeline. | source-specific / relative |
| `river_discharge_m3s` | Estimated daily river discharge/flow from GloFAS reanalysis via Open-Meteo Flood API; **not** a direct observed river water level. | m³/s |

### Source missingness flags

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `rainfall_local_mm_missing` | Boolean flag indicating that the corresponding original local rainfall input was missing/unavailable at the source or pipeline stage. A numeric filled value does not by itself mean an original observation existed. | True / False |
| `rainfall_upstream_mm_missing` | Boolean flag indicating that the corresponding original upstream rainfall input was missing/unavailable at the source or pipeline stage. A numeric filled value does not by itself mean an original observation existed. | True / False |
| `soil_moisture_local_missing` | Boolean flag indicating that the corresponding original local soil moisture input was missing/unavailable at the source or pipeline stage. A numeric filled value does not by itself mean an original observation existed. | True / False |
| `river_discharge_m3s_missing` | Boolean flag indicating that the corresponding original river discharge input was missing/unavailable at the source or pipeline stage. A numeric filled value does not by itself mean an original observation existed. | True / False |
| `flood_byStor_missing` | Boolean flag indicating that the corresponding original original `flood_byStor` source indicator input was missing/unavailable at the source or pipeline stage. A numeric filled value does not by itself mean an original observation existed. | True / False |

### Lagged environmental variables

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `rainfall_local_mm_lag1d` | Local rainfall **1 day(s) before** the record date (`t-1`); historical feature.  | mm |
| `rainfall_local_mm_lag2d` | Local rainfall **2 day(s) before** the record date (`t-2`); historical feature.  | mm |
| `rainfall_local_mm_lag3d` | Local rainfall **3 day(s) before** the record date (`t-3`); historical feature.  | mm |
| `rainfall_local_mm_lag5d` | Local rainfall **5 day(s) before** the record date (`t-5`); historical feature.  | mm |
| `rainfall_upstream_mm_lag1d` | Upstream rainfall **1 day(s) before** the record date (`t-1`); historical feature.  | mm |
| `rainfall_upstream_mm_lag2d` | Upstream rainfall **2 day(s) before** the record date (`t-2`); historical feature.  | mm |
| `rainfall_upstream_mm_lag3d` | Upstream rainfall **3 day(s) before** the record date (`t-3`); historical feature.  | mm |
| `rainfall_upstream_mm_lag5d` | Upstream rainfall **5 day(s) before** the record date (`t-5`); historical feature.  | mm |
| `soil_moisture_local_lag1d` | Local soil moisture **1 day(s) before** the record date (`t-1`); historical feature.  | source-specific / relative |
| `soil_moisture_local_lag2d` | Local soil moisture **2 day(s) before** the record date (`t-2`); historical feature.  | source-specific / relative |
| `soil_moisture_local_lag3d` | Local soil moisture **3 day(s) before** the record date (`t-3`); historical feature.  | source-specific / relative |
| `soil_moisture_local_lag5d` | Local soil moisture **5 day(s) before** the record date (`t-5`); historical feature.  | source-specific / relative |
| `river_discharge_m3s_lag1d` | Estimated river discharge **1 day(s) before** the record date (`t-1`); historical feature.  | m³/s |
| `river_discharge_m3s_lag2d` | Estimated river discharge **2 day(s) before** the record date (`t-2`); historical feature.  | m³/s |
| `river_discharge_m3s_lag3d` | Estimated river discharge **3 day(s) before** the record date (`t-3`); historical feature.  | m³/s |
| `river_discharge_m3s_lag5d` | Estimated river discharge **5 day(s) before** the record date (`t-5`); historical feature.  | m³/s |

### Rolling & derived variables

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `rainfall_local_mm_sum7d` | Local rainfall accumulated over a rolling 7-day window; exact boundary convention follows the source feature pipeline. | mm |
| `rainfall_local_mm_sum14d` | Local rainfall accumulated over a rolling 14-day window; exact boundary convention follows the source feature pipeline. | mm |
| `rainfall_upstream_mm_sum7d` | Upstream/catchment rainfall accumulated over a rolling 7-day window; exact boundary convention follows the source feature pipeline. | mm |
| `rainfall_upstream_mm_sum14d` | Upstream/catchment rainfall accumulated over a rolling 14-day window; exact boundary convention follows the source feature pipeline. | mm |
| `soil_moisture_delta_30d` | Difference/change in local soil-moisture level relative to a 30-day historical reference; not an absolute moisture reading. | source-specific / relative |
| `soil_moisture_swi` | Derived Soil Water Index, summarizing antecedent wetness/soil-water conditions; see source feature engineering for the exact formula. | index / dimensionless |

### Terrain

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `elevation_m` | Approximate terrain elevation of the station/site, derived from a digital elevation model. | m |
| `hand_m` | Height Above Nearest Drainage (HAND): terrain elevation relative to the nearest drainage channel; static topographic exposure proxy, not current river level. | m |

### Upstream river network

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `upstream_chain_discharge_lag1d` | One-day-lagged estimated discharge from a candidate upstream station in the river network. Available only where network linkage/data exist; mostly missing in the cleaned multi-station CSV. | m³/s |

### Flood labels

| Column (exact name) | Meaning | Units / values |
|---|---|---|
| `flood_within_24h` | Forward-looking binary label for whether flooding was reported/identified within the next 24 hours after the row date; `1` = positive, `0` = negative, missing = unavailable in the full source. Label quality depends on source coverage. | 0 / 1 / missing |
| `flood_within_24h_label_regime` | Provenance/coverage category for the 24-hour label. The cleaned 2021–2025 subset uses `observed`; this does **not** guarantee independent confirmation of each individual label. | category |
| `flood_within_48h` | Forward-looking binary label for whether flooding was reported/identified within the next 48 hours after the row date; `1` = positive, `0` = negative, missing = unavailable in the full source. Label quality depends on source coverage. | 0 / 1 / missing |
| `flood_within_48h_label_regime` | Provenance/coverage category for the 48-hour label. The cleaned 2021–2025 subset uses `observed`; this does **not** guarantee independent confirmation of each individual label. | category |
| `flood_within_72h` | Forward-looking binary label for whether flooding was reported/identified within the next 72 hours after the row date; `1` = positive, `0` = negative, missing = unavailable in the full source. Label quality depends on source coverage. | 0 / 1 / missing |
| `flood_within_72h_label_regime` | Provenance/coverage category for the 72-hour label. The cleaned 2021–2025 subset uses `observed`; this does **not** guarantee independent confirmation of each individual label. | category |

### Original columns not retained in the clean CSV

The **49-column original Parquet** also includes the following **six** variables removed before export (the cleaned file separately adds `year` and `basin`, giving 45 columns):

| Original column | Meaning / why not carried into the cleaned CSV |
|---|---|
| `flood_byStor` | Very sparse source-specific flood indicator in the upstream dataset; exact source encoding is not established in this project. Removed from EDA subset (its `_missing` flag remained). |
| `rainfall_local_mm_trend_ratio` | Engineered ratio comparing recent rainfall to an earlier period; missingness and denominator issues in some records. |
| `rainfall_upstream_mm_trend_ratio` | Engineered upstream rainfall trend ratio; removed during missing-value cleaning. |
| `upstream_chain_discharge_lag2d` | Two-day-lagged network/upstream discharge; sparse availability. |
| `upstream_reference_discharge_lag2d` | Two-day-lagged discharge of a designated upstream reference station; sparse availability. |
| `upstream_reference_discharge_lag3d` | Three-day-lagged upstream reference discharge; sparse availability. |

No columns have been renamed to shorter aliases in the data dictionary; `year` and `basin` are extra columns introduced during this project's EDA preparation.

## 5. Exploratory Data Analysis (EDA)

The EDA notebook documents: raw dataset structure and missingness; observed versus unavailable labels; distributions of environmental variables; flooding class imbalance; station-level and river-basin comparisons; correlations; local versus upstream predictor checks; and seasonality of positive labels. Stations SW90 and SW17 were then selected for a simple controlled comparison.

**Key EDA observations:**

- The full historical file contains far more dates than reliably available flood labels. Selecting an observed-label subset matters for supervised learning.
- Positive flood labels are substantially rarer than non-flood labels, particularly at SW90 and SW17. High classification accuracy can therefore be misleading.
- Rainfall, soil moisture and discharge vary seasonally; however, a seasonal rule alone produces many false positive predictions.
- Two stations can show similar discharge but different local rain/soil-moisture conditions. Adding SW17 data is therefore a testable hypothesis, **not** a guaranteed improvement.

## 6. Feature engineering: three-day sliding-window chunks

The modeling target is **`flood_within_24h` for SW90**. In the modeling notebook, features use consecutive days `t-2`, `t-1`, and `t`, and the source forward-looking label at `t` is interpreted as risk for the **following day** (`t+1`). The target is not constructed from a manually chosen discharge threshold.

| Example | Input features available by end of day | Prediction date |
|---|---|---|
| Chunk 1 | Jan 1, 2, 3 | Jan 4 |
| Chunk 2 | Jan 2, 3, 4 | Jan 5 |
| Chunk 3 | Jan 3, 4, 5 | Jan 6 |

The windows are overlapping, not three-day non-overlapping blocks. Models receive 1 row **per three-day window** rather than one row per individual input day. The next-day label alignment relies on the original dataset's documented forward-looking definition; the exact daily observation cutoff and flood-label timestamp should be verified before operational use.

**Experiment features:**

| Experiment | Data supplied for each of the 3 days | Feature count | Target |
|---|---|---:|---|
| Logistic Regression A | SW90 local rainfall, local soil moisture, river discharge | **9** | SW90 next-24h flood label |
| Logistic Regression B | Same three variables from **both SW90 and SW17** | **18** | SW90 next-24h flood label |
| Logistic Regression C (exploratory) | All 9 from SW90 + SW17 river discharge on days 1, 2, 3 | **12** | SW90 next-24h flood label |
| Random Forest (main) | Same 18 variables as model B | **18** | SW90 next-24h flood label |
| Random Forest + upstream rainfall (exploratory) | Same 18 variables + `rainfall_upstream_mm` for days 1, 2, 3 | **21** | SW90 next-24h flood label |

**Important:** SW17 contributes **past/current-day environmental measurements only**. The models do **not** use SW17's future flood label as a predictor. Precomputed lag columns from the source file were not used to build the principal sliding windows: instead, the notebook assembled three consecutive calendar days explicitly.

## 7. Data split and evaluation procedure

For the primary 2021–2024 experiments, **2025 is excluded**, leaving **1,458** consecutive 3-day windows with prediction dates between Jan 4, 2021 and Dec 31, 2024. Data are sorted by prediction date, then split **chronologically** rather than shuffled: the oldest **1,020 windows (70%)** train the model and the newest **438 windows (30%)** form the test set (from late October 2023 through 2024).

The held-out test set contains **431 negative** and only **7 positive** 24-hour SW90 flood labels. Logistic Regression uses a `StandardScaler` fitted on training data only. Random Forest uses raw numerical features without scaling. Extra experiments tried `class_weight='balanced'`, probability thresholds and `max_depth=5`.

Metrics reported for the **positive (flood) class**:

- **Accuracy** — fraction of all predictions correct; dominated by frequent no-flood days.
- **Precision** — of the predicted flood days, fraction labeled flood; low precision means many false alarms.
- **Recall** — of the labeled flood days, fraction detected; low recall means missed flood days.
- **F1-score** — harmonic mean of positive-class precision and recall.
- **Confusion matrix** — numbers of true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP).

A temporary chronological **train/validation split** within the training portion was also explored. Since different settings were examined after viewing the same 2024 test results, the reported test numbers should be understood as **exploratory** and not as a pristine final generalization estimate.

## 8. Seasonal baseline (June–August)

The intended baseline makes a calendar-only prediction for each **target date**:

```python
seasonal_baseline = test_rf['target_date'].dt.month.isin([6, 7, 8]).astype(int)
```

This predicts `1` for every day in **June, July or August**, regardless of rainfall, discharge or moisture, and `0` otherwise. It is deliberately simple, but provides a concrete comparison against the main models.

## 9. Model results — same 438 test windows

| Method | Test accuracy | Flood precision | Flood recall | Flood F1 | Flood days detected | False alarms |
|---|---:|---:|---:|---:|---:|---:|
| **Seasonal baseline** (Jun/Jul/Aug = 1) | 79.68% | 5.43% | **71.43%** | 10.10% | **5 / 7** | 87 |
| **Logistic Regression — SW90 only** | **98.63%** | 66.67% | 28.57% | **40.00%** | 2 / 7 | 1 |
| Logistic Regression — SW90 + SW17 | 97.95% | 25.00% | 14.29% | 18.18% | 1 / 7 | 3 |
| Random Forest — SW90 + SW17 (default) | 98.17% | 33.33% | 14.29% | 20.00% | 1 / 7 | 2 |
| Random Forest — SW90 + SW17 (`max_depth=5`) | **98.63%** | **100.00%** | 14.29% | 25.00% | 1 / 7 | **0** |
| Random Forest — 21 features (`max_depth=5`) | **98.63%** | **100.00%** | 14.29% | 25.00% | 1 / 7 | **0** |

**Interpretation:** The SW90-only Logistic Regression has the highest positive-class F1 among the listed test experiments (**0.40**) and it detects two of the seven positive days. The depth-limited Random Forest generated no false alarms, but correctly flagged only one positive day; its `100%` flood precision is based on **one** positive prediction, so it is not evidence of reliable precision in general. The seasonal baseline detects more positive days (**5/7**) at the cost of **87 false alarms**.

A separate **always-no-flood** rule would score **431/438 = 98.40% accuracy**, higher than the seasonal baseline and close to the trained models. For this reason, high accuracy alone must not be presented as proof of a useful flood warning system.

On one chronological **validation** period, SW90+SW17 Logistic Regression detected **5 of 11** positive labels versus **1 of 11** for SW90 alone. Four extra detected days were consecutive in June 2023 and may correspond to **one event**. The additional-station benefit **did not carry over** to the examined test set. This is an example of the uncertainty of evaluation with rare, temporally clustered events.

Testing `class_weight='balanced'` on SW90+SW17 Logistic Regression increased validation recall (**8/11**) but produced **36 false alarms** out of 204 validation windows. Trying alternative probability thresholds did not yield a clear improvement over the standard model at threshold `0.5`. Limiting RF tree depth reduced false positives but did not increase test flood recall. Adding three upstream-rainfall values (21 vs. 18 features) did **not change the binary predictions on any of the 438 test dates**; it is not evidence that upstream rain is intrinsically irrelevant.

## 10. Data quality investigation (no target relabeling)

I reviewed selected disagreements between the original SW90 labels and Random Forest predictions, and separately reviewed dates with high estimated river flow and independent flood reporting. Examples:

| SW90 prediction date | Original label | Depth-limited RF prediction | Review note |
|---|---:|---:|---|
| Mar 21, 2024 | 1 | 0 | Low estimated discharge in preceding three-day window; not enough evidence to call the source label wrong. |
| Jun 19, 2024 | 1 | 0 | Substantial estimated discharge and upstream rainfall; model's miss may be genuine. |
| Jul 4, 2024 | 0 | 0 | News reports describe flooding in the surrounding Jamalpur district and water above danger level at Bahadurabad. |
| Jul 5, 2024 | 0 | 0 | Continued local flood reporting raises a **review question**, not independent proof that the target definition was violated. |

Relevant reports from [Bangladesh Sangbad Sangstha (BSS), July 4, 2024](https://www.bssnews.net/district/198417) and [July 5, 2024](https://www.bssnews.net/district/198642) document flooding in parts of Jamalpur and above-danger river stage at Bahadurabad. These reports use human and river-stage measures; the model's target comes from an aggregated flood-label pipeline, so the definitions and timing **may not match exactly**.

No original `flood_within_24h` labels were changed. The separate notebook audit field `review_status` is **an annotation for investigation**, not a new verified target. It would be inappropriate to change `0` to `1` based only on high discharge, or to erase a `1` because a model predicts `0`.

## 11. Main limitations and caution

1. **Severe class imbalance:** just **7 positive days in the 438-day test set**, leading to unstable precision/recall estimates and a misleadingly high accuracy for no-flood predictions.
2. **Overlapping windows & correlated days:** adjacent chunks share two of their three input days; sequential positive days may be one flood episode. Per-day scores overstate the number of independent events.
3. **Model/label definition uncertainty:** positive labels are combined from several remote-sensing/event sources with variable coverage, while negative labels are not guaranteed proof of no flood. Reports of above-danger river stage do not exactly equal the source's binary flood extent definition.
4. **Reanalysis vs. ground observations:** daily estimated river discharge is **not** measured water level. Rainfall and soil moisture also come from reanalysis-based products.
5. **Time alignment:** this is a next-24h, forward-looking target. The code maps the row-date label at the end of a 3-day chunk to the following date; precise observation/report cutoffs must be validated before any operational forecasting.
6. **Potential upstream linkage limits:** similar flow series at SW17 and SW90 do not prove independent predictive information or correct travel-time alignment. Extra station features may add redundancy.
7. **Limited temporal testing:** one chronological 70/30 split, with 2025 excluded; model performance may change for other monsoon seasons or future years.
8. **Exploratory tuning / test reuse:** multiple choices were explored after examining the same test results; a future study needs fresh later test data or walk-forward validation for an unbiased final score.
9. **Threshold and class-weight trade-offs:** a model can detect more positives by issuing many more false alarms. Practical warning costs were not optimized or calibrated.
10. **Sparse network features:** the cleaned full-station file retains an upstream discharge lag with ~90% missingness; main models therefore used explicitly assembled SW90/SW17 measurements rather than this sparse column.
11. **Not suitable for operational alerting:** no field deployment, alert calibration, independent event-level verification or safety validation has been performed.

## 12. Reproducing the notebook approach

1. Open the project's EDA notebook and review original `df`, cleaning, label coverage and station choice.
2. Obtain the prepared **`Data_Bangaldesh_Clean.csv`** (the exact spelling is part of this project), and load it with `pandas.read_csv`.
3. Convert `date` to datetime and isolate `station_id` = `SW90` and `SW17`, sorting dates ascending.
4. Join SW90's flood target and measurements with **only measurements** from SW17 on the same date.
5. Build consecutive **three-day** sliding windows; verify no gaps/missing values; retain `target_date` and `flood_tomorrow` as non-feature fields.
6. Keep windows targeting dates through Dec 31, 2024, make a **chronological** 70/30 split and fit Logistic Regression (`StandardScaler` fitted on train) and Random Forest (no scaler).
7. Evaluate on the same windows, compute the June–August seasonal baseline and compare Precision, Recall, F1, confusion matrices and false alarms.
8. Review data-quality anomalies separately and **do not overwrite** the source target based on model disagreements.

Suggested notebook names (actual location may vary by repository organization):

- EDA: `CapStoneProjectJankoBurzanovicEDA_Explained.ipynb`
- Modeling: `Modeling_Capstone_Project_Janko.ipynb` (and/or the annotated edition)

### Minimal dependencies

- Python 3, pandas, NumPy
- Matplotlib, Seaborn, missingno (EDA)
- scikit-learn (classification and evaluation)
- A Parquet reader such as pyarrow for the original `.parquet` EDA file

## 13. Conclusions

This project demonstrates an end-to-end, transparent flood-prediction experiment built around **three-day sliding windows** and simple classifiers. The measured outcome is not an operational forecasting system: although some models outperform the June–August seasonal baseline in **Accuracy and positive-class F1**, **none consistently identifies the majority of labeled flood days**. The seasonal baseline detects more flood days but issues many false alarms. Among the reported test models, **SW90-only Logistic Regression** gives the strongest **Flood F1 (0.40)**, while adding SW17 or upstream rainfall does not reliably improve generalization on the observed test period.

**Main learning:** successful Data Science is not just a high score. It requires a time-aware split, an explicit baseline, evaluation of the rare class, careful feature timing, and an honest assessment of data/label limitations. A future extension could confirm flood episodes against independent station-level evidence, evaluate performance by distinct flood events, and test on a genuinely later monsoon season.

## 14. References

- [Source project — Bangladesh Flood Forecasting Dataset](https://github.com/Srabon-bot/Bangladesh-flood-data-set)
- [Source dataset file (Parquet)](https://github.com/Srabon-bot/Bangladesh-flood-data-set/tree/main/data/features/2026-08-07c)
- [Source feature-building and station metadata code](https://github.com/Srabon-bot/Bangladesh-flood-data-set/tree/main/train)
- [BSS: Flood affects two upazilas in Jamalpur, July 4, 2024](https://www.bssnews.net/district/198417)
- [BSS: Flood situation deteriorates in Jamalpur, July 5, 2024](https://www.bssnews.net/district/198642)

---

*Prepared for a Data Science Capstone. All stated metrics refer to the notebook's reported exploratory results, not to an independently deployed flood-warning system.*
