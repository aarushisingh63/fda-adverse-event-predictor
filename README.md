# Predicting the Seriousness of FDA Drug Adverse Event Reports (FAERS, 2015-2025)

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
**Data:** [FDA FAERS 2015-2026 on Kaggle](https://www.kaggle.com/datasets/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers). The reports in this file cover 2015-01-01 to 2025-12-31.

---

## Problem statement

Every year the FDA receives a very large number of reports of side effects from drugs. Some describe a mild reaction, but others describe a patient who was hospitalized, whose life was in danger, who was left with a disability, or who died. Reports are reviewed manually, and there is no automatic way to tell at the time of submission which ones are serious and should be looked at first.

This project asks a simple question:

> Can the seriousness of an adverse event report be predicted from the information that is available when the report is submitted?

---

## What was done in this project

**1. Getting to know the data.** I started with 528,000 reports and 30 variables. 74.81% of the reports are serious, so a model that always answers "serious" is already correct 74.81% of the time. I used this as the baseline that any real model must beat. I then looked at how the serious rate changes with age, number of drugs, number of reactions, sex, country, drug, month and year.

**2. Finding out what the data can and cannot tell.** Three things stood out:

- Every year from 2015 to 2025 has exactly 48,000 reports. The data were sampled evenly by year, so the number of reports says nothing about real trends and I did not read any trend into it.
- Missing information is not random. Reports with a missing age are serious 65.4% of the time against 78.6% when age is recorded, so "missing" became a variable of its own.
- The reporting country matters a lot: 59.9% of US reports are serious against 97.3% of non-US reports.

**3. Preparing the data carefully.** Because the model is meant to work when a report arrives, I kept only information available at that time, so the columns that describe the outcome itself were left out (details in Procedure, step 2). I split the data into 80% for training and 20% for testing before doing anything else, and learned all preprocessing values (medians, category lists, scaling) from the training part only.

**4. Building and comparing models.** I fitted Logistic Regression and Random Forest. Since the country could be giving the models an easy shortcut, I fitted each of them three times: with all variables, without country, and without country and drug name. This shows how much of the performance depends on them.

**5. Checking where the models go wrong.** Besides accuracy, I looked at precision, recall, F1, AUC and confusion matrices, and at which variables the Random Forest used most.

**What came out of it:** the best model (Random Forest, all variables) reached an accuracy of 84.58% and an AUC of 0.911, about 10 points above the baseline accuracy. Without country the AUC falls to 0.867, so country carries a large part of the signal, but the model still clearly beats the baseline. The number of reactions and the number of drugs were the most useful predictors. The reason for the country effect could not be established from this data, and the project has clear limitations (single split, no tuning, sampled data), which are listed at the end.

---

## Repository contents

| File | Content |
|---|---|
| `faers_seriousness_prediction.ipynb` | Part 1: exploratory analysis. Part 2: preprocessing. Part 3: modelling and evaluation. Most code cells are followed by an interpretation, and the charts are shown in the notebook outputs. |
| `README.md` | This file |

---

## Procedure

### 1. Data audit and exploratory analysis

The data contain 528,000 reports and 30 variables, with no duplicate report IDs. Weight is missing in 72.0% of the reports and age in 28.7%.

| Variable | Observation (raw serious rate) |
|---|---|
| Overall | 395,000 serious (74.81%), 133,000 not serious (25.19%), a class ratio of about 3:1 |
| Age group | Infants (0-2) 90.3%, elderly (81+) 81.1%, children (3-12) 67.8%, teens 72.6% (the infant group has 3,101 reports) |
| Number of drugs | Single drug 58.8%, 2-3 drugs 67.3%, 4-5 drugs 77.6%, 6 or more 85.7% |
| Number of reactions | About 68-71% up to 5 reactions, then 80.7% (6-10), 92.0% (11-30), 98.8% (more than 30): a non-linear relationship |
| Missing age | 65.4% serious when missing, 78.6% when recorded |
| Missing weight | 71.7% serious when missing, 82.7% when recorded |
| Month, year | No seasonal pattern (71-79% by month) and no trend (68-82% by year) |
| Linear correlation with response | `num_reactions` 0.178, `num_drugs` 0.126, age 0.019, weight -0.029 |

The most reported drugs are Tofacitinib, Risperidone and Rivaroxaban.

**Sampling artifact.** Every year from 2015 to 2025 has exactly 48,000 reports, so the data were sampled evenly by year. Report volume therefore carries no information about real trends, and no trend conclusions were drawn from it.

### 2. Data leakage audit

A predictor must be known when the prediction is made. Ten variables were removed:

- **Outcome variables:** `serious_flags`, `is_fatal`, `is_hospitalized`, `is_life_threat`, `is_disabling`, `reaction_outcomes`, `patient_recovered`.
- **Reaction text:** `reactions`, `primary_reaction`. Every report with "Death" as primary reaction is serious (100%), and 2.72% of all reports mention "Death" in the reaction list.

Seven further variables were dropped because they are identifiers or redundant: `report_id`, `receive_date`, `report_age_days`, `quarter`, `year`, `month` and `brand_name`.

### 3. Reporting country

| | Reports | Serious rate |
|---|---|---|
| US | 317,456 | 59.9% |
| Non-US | 210,544 | 97.3% |

Among the ten countries with most reports, every named non-US country or region (France, EU, UK, Italy, Japan, Brazil, Canada, Germany) is at 97-99%, while reports with an unknown country are at about 85%. A drug-wise check of the ten highest-rate drugs shows two patterns: some are almost entirely non-US (Jakavi 0.1% US reports, Prednisolone 5.1%, Atezolizumab 8.6%), so their rate of about 99% is largely a country effect; others are mostly US reports (Zantac 99.5% US, Rivaroxaban 95.1%, Lansoprazole 88.7%) and are also about 99% serious, which the country mix does not explain. The cause of the country difference could not be established from the data. A reporting rule is a possible explanation, but this is only a hypothesis.

### 4. Preprocessing

All quantities were estimated from the **training set only**, after the split.

| Step | Method |
|---|---|
| Split | Stratified 80/20 split (422,400 training and 105,600 test reports, `random_state=42`); the serious rate is 74.81% in both sets |
| Missing age and weight | Training-set median; indicator variables `age_missing` and `weight_missing` keep the information that a value was missing |
| High-cardinality categories | The 20 most frequent categories of `suspect_drug`, `manufacturer`, `pharm_class`, `drug_indication`, `country` and `drug_route` are kept, the rest are grouped as "Other" (71.2% of `suspect_drug` values become "Other") |
| Skewed counts | `log(1 + x)` for `num_drugs` (maximum 2,108) and `num_reactions` (maximum 253) |
| Encoding | One-hot encoding with one category dropped per variable to avoid perfect multicollinearity; route codes treated as categories; 139 columns |
| Scaling | Standardisation of the four numeric variables with training-set mean and standard deviation |

### 5. Models and ablation

- **Logistic Regression** (`max_iter=1000`, default regularisation of scikit-learn) as a linear baseline.
- **Random Forest** (100 trees, maximum depth 20, `random_state=42`), able to capture non-linear relationships.

Both were fitted without class weights and evaluated on the same test set with a cut-off of 0.5. Each model was fitted on three predictor sets:

- **A:** all 139 columns
- **B:** without the country columns (119 columns)
- **C:** without the country and suspect-drug columns (99 columns)

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| A - Logistic Regression | 83.73% | 87.72% | 90.99% | 89.33% | 0.897 |
| **A - Random Forest** | **84.58%** | 86.95% | 93.41% | **90.06%** | **0.911** |
| B - Logistic Regression | 81.38% | 83.76% | 93.18% | 88.22% | 0.841 |
| B - Random Forest | 82.40% | 83.35% | 95.57% | 89.04% | 0.867 |
| C - Logistic Regression | 80.92% | 83.35% | 93.09% | 87.95% | 0.834 |
| C - Random Forest | 82.42% | 83.34% | 95.62% | 89.06% | 0.868 |

Baseline (always predict "serious"): accuracy 74.81%.

1. All six models are clearly better than the baseline.
2. Random Forest is better than Logistic Regression in every version, but the AUC gap is modest (0.014, 0.026 and 0.034 in A, B and C).
3. Removing country causes the largest drop (Random Forest AUC 0.911 to 0.867, Logistic Regression 0.897 to 0.841). Part of the headline performance comes from the country effect.
4. Removing the drug name as well changes almost nothing (Random Forest AUC 0.867 to 0.868).
5. Without country the models are still well above the baseline, so performance does not rest on country alone. Versions B and C are the more realistic results.

### Error analysis

The test set has 79,000 serious and 26,600 non-serious reports. Errors at the cut-off of 0.5:

| Model | Serious reports missed | Non-serious reports flagged as serious |
|---|---|---|
| A - Random Forest | 5,205 (6.6%) | 11,077 of 26,600 (41.6%) |
| A - Logistic Regression | 7,121 (9.0%) | 10,058 (37.8%) |
| B - Random Forest | 3,503 (4.4%) | 15,085 (56.7%) |
| B - Logistic Regression | 5,390 (6.8%) | 14,271 (53.7%) |

All models find serious reports better than they recognise non-serious ones. Without country, the Random Forest finds 95.6% of the serious reports but flags 56.7% of the non-serious ones, so it leans towards predicting "serious". For a review-priority tool a missed serious report is the costly error, so this direction is safer, but it creates many unnecessary reviews.

### Feature importance (Random Forest, Version B)

The most important variables are `num_reactions` (0.112), `num_drugs` (0.083), `patient_age_years` (0.047), the 6+ drugs category (0.045) and the drug Risperidone (0.040), which agrees with the exploratory analysis. `age_missing` also appears, supporting the finding that missingness carries information. Importance shows which variables were used, not the direction of an effect, and it does not prove causation.

---

## Limitations

- A **single train/test split** was used, with no cross-validation, confidence intervals or hyperparameter tuning. Small differences between models (for example 0.014 in AUC) should not be over-interpreted.
- The **cause of the country effect is unknown**. Version B removes only the country columns, and other variables may still carry country-related information. The descriptive rates above are not adjusted for country.
- The data are a **sample** with exactly 48,000 reports per year, so the 74.81% serious rate does not describe the full FAERS database.
- `num_reactions` may partly reflect more detailed documentation of serious cases.
- Only the top 20 categories of high-cardinality variables were kept, so rare drugs, manufacturers and countries are not modelled individually.
- The cut-off of 0.5 was **not tuned** and the model was not tested on new data. This is a proof of concept and not a validated clinical or regulatory tool.

---

## How to run

1. Download `faers_seriousness_prediction.ipynb` and open it in Jupyter Notebook or JupyterLab.
2. Install the libraries if needed: `pip install pandas numpy scikit-learn matplotlib seaborn kagglehub pyarrow`
3. Run all cells from top to bottom (Jupyter: **Kernel, Restart and Run All**). The data are downloaded with `kagglehub`; if Kaggle asks for credentials, follow the instructions shown on the screen.

The notebook already contains all outputs and charts, so it can be read on GitHub without running it. The processed data files created in `data/` are not included in the repository.

---

## Author

Aarushi Singh Rajput, MSc Statistics
