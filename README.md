# Traverlers Insurance Claim Fraud Detection

![](https://i.imgur.com/QffJHZY.gif)

This project was originally a modeling competition for the class STAT 8501 Advancned Regression Technique, hosted by [Travelers Insurance](https://www.travelers.com). The task was to identify first party physical damage fraudulence. The rules of this competition can be found in [Kaggle](https://www.kaggle.com/c/2018-trv-statistical-modeling-competition-umn). My team ranked 2nd place out of 8 teams. The code my team used was quite messy and not really reproducible. I completely re-organized it and deployed it as a Flask app.

## Directory Structure

```
.
├── LICENSE
├── README.md
├── data                           <- Datasets.
│   ├── external                       <- Data from third party sources.
│   ├── processed                      <- The final, canonical data sets for modeling.
│   └── raw                            <- The original, immutable data dump.
├── models                         <- Trained and serialized models.
│   └── best_model.pickle
├── notebooks                      <- Jupyter notebooks.
│   ├── 1-data-exploration.ipynb       <- Explore data with histograms, correlation matrix, etc.
│   ├── 2-data-preprocessing.ipynb     <- Data cleaning, feature engineering.
│   └── 3-modeling.ipynb               <- Cross validation, hyperparameter tuning.
├── requirements.txt
└── src                            <- Source code for the web app.
    ├── app.py                         <- Flask API.
    ├── templates                      <- HTML files.
    └── utils.py                       <- Utilities functions for handling form data from app.
```

## Project Overview

The training set has 18,000 samples and the test set has 12,000 samples. The following variables are available:

- claim_number - Claim ID (cannot be used in model)
- age_of_driver - Age of driver
- gender - Gender of driver
- marital_status - Marital status of driver
- safty_rating - Safety rating index of driver
- annual_income - Annual income of driver
- high_education_ind - Driver’s high education index
- address_change_ind - Whether or not the driver changed living address in past 1 year
- living_status - Driver’s living status, own or rent
- zip_code - Driver’s living address zipcode
- claim_date - Date of first notice of claim
- claim_day_of_week - Day of week of first notice of claim
- accident_site - Accident location, highway, parking lot or local
- past_num_of_claims - Number of claims the driver reported in past 5 years
- witness_present_ind - Witness indicator of the claim
- liab_prct - Liability percentage of the claim
- channel - The channel of policy purchasing
- policy_report_filed_ind - Policy report filed indicator
- claim_est_payout - Estimated claim payout
- age_of_vehicle - Age of first party vehicle
- vehicle_category - Category of first party vehicle
- vehicle_price - Price of first party vehicle
- vehicle_color - Color of first party vehicle
- vehicle_weight - Weight of first party vehicle
- fraud - Fraud indicator (0=no, 1=yes). This is the response variable

The goal is to predict `fraud` with other variables. The evaluation metric is AUC.

I used a 5-fold cross-validation to compare the performance of k-nearest neighbors, random forest, logistic regression and gradient boosting. Gradient boosting had the best performance, so I did some hyperparameter tuning on it using random searh. The final model had an AUC of 0.74 on the test set, and was deployed as a web app on an AWS EC2 Instance.
