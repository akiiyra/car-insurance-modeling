# Car Insurance Claim Prediction

Identifying the single best predictor of insurance claims using univariate logistic regression, completed as part of a DataCamp project.

## Overview

On the Road car insurance wants to predict whether a customer will make a claim during their policy period. They have limited ML infrastructure, so rather than a complex multi-feature model, they need the **single feature** that produces the best-performing model by accuracy — something simple enough to deploy and monitor immediately.

This project builds one logistic regression model per feature, evaluates each, and identifies the winner.

## Dataset

`car_insurance.csv` — client demographics, driving history, and vehicle details, with a binary `outcome` column indicating whether a claim was made.

| Column | Description |
|---|---|
| `id` | Unique client identifier |
| `age` | Age bracket (0: 16–25, 1: 26–39, 2: 40–64, 3: 65+) |
| `gender` | 0: Female, 1: Male |
| `driving_experience` | Years driving (0: 0–9, 1: 10–19, 2: 20–29, 3: 30+) |
| `education` | 0: None, 1: High school, 2: University |
| `income` | 0: Poverty, 1: Working class, 2: Middle class, 3: Upper class |
| `credit_score` | Score between 0 and 1 |
| `vehicle_ownership` | 0: Financing, 1: Owns outright |
| `vehicle_year` | 0: Before 2015, 1: 2015 or later |
| `married` | 0: Not married, 1: Married |
| `children` | Number of children |
| `postal_code` | Client's postal code |
| `annual_mileage` | Miles driven per year |
| `vehicle_type` | 0: Sedan, 1: Sports car |
| `speeding_violations` | Total speeding violations |
| `duis` | Times caught driving under the influence |
| `past_accidents` | Total previous accidents |
| `outcome` | **Target** — 0: No claim, 1: Made a claim |

## Approach

1. **Exploration** — inspected structure, dtypes, and summary statistics
2. **Missing values** — `credit_score` and `annual_mileage` contained nulls; imputed with column means
3. **Feature setup** — dropped `outcome` (target) and `id` (identifier), leaving 16 candidate features
4. **Modelling** — fitted a separate `statsmodels` logistic regression for each feature using formula syntax (`outcome ~ feature`)
5. **Evaluation** — derived accuracy from each model's confusion matrix via `pred_table()`:

accuracy = (TN + TP) / (TN + FP + FN + TP)

6. **Selection** — returned the feature with the highest accuracy as a single-row DataFrame

## Result

**`driving_experience`** was the strongest single predictor, achieving roughly **77.7% accuracy**.

Intuitive: how long someone has been driving carries more signal about claim likelihood than age, income, or vehicle type.

## Tech Stack

- Python
- pandas
- NumPy
- statsmodels
