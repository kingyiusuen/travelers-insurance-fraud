# Traverlers Insurance Claim Fraud Detection

This project was originally a modeling competition for the class STAT 8501 Advancned Regression Technique, hosted by [Travelers Insurance](https://www.travelers.com). The task was to identify first party physical damage fraudulence. The rules of this competition can be found in [Kaggle](https://www.kaggle.com/c/2018-trv-statistical-modeling-competition-umn).

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
│   ├── 1-eda.ipynb                    <- Explore data with histograms, correlation matrix, etc.
│   ├── 2-data_preprocessing.ipynb     <- Data cleaning, feature engineering.
│   └── 3-modeling.ipynb               <- Cross validation, hyperparameter tuning.
├── requirements.txt
└── src                            <- Source code for the web app.
    ├── app.py                         <- Flask API.
    ├── templates                      <- HTML files.
    └── utils.py                       <- Utilities functions for handling form data from app.
```

## Project Overview

### Data

The training set has 18,000 samples and the test set has 12,000 samples. The following variables are available:

- claim_number - Claim ID (cannot be used in model)
- age_of_driver - Age of driver
- gender - Gender of driver
- marital_status - Marital status of driver
- safty_rating - Safety rating index of driver
- annual_income - Annual income of driver
- high_education_ind - Driver's high education index
- address_change_ind - Whether or not the driver changed living address in past 1 year
- living_status - Driver's living status, own or rent
- zip_code - Driver's living address zipcode
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

The goal is to predict `fraud` with other variables. I doubted Travelers provided a real dataset for us, because about 15% of the observations had `fraud` = 0. I believe that there should be a class imbalance in real life and the percentage of fraudulent claims should be much small.

### Data Cleaning and Feature Engineering

The following preprocessing steps were applied:

- Remove observations with invalid values (`fraud` = -1)
- Remove features that are clearly not useful e.g., `vehicle_color`
- Replace outliers (e.g., `age_of_driver` > 100) with mean
- Mean/mode imputation for missing values
- Dummify categorical features
- Normalize the feature with min-max scaling
- Since there are many unique values for `zip_code`, I experimented with target encoding and transforming it into `latitude` and `longitude`

### Modeling

Since the test set provided by the host does not include the target variable, I used 20% of the training set as a validation set for model evaluation.The evaluation metric was AUC. Models that were examined included k-nearest neighbors (kNN), logistic regression and XGBoost. For kNN, a grid search was used to find the optimal value of $k$. For XGBoost, the number of hyperparameters was larger, so a random search was used for hyperparameter tuning instead. The best performing model was XGBoost, which achieved an AUC of 0.73 on the validation set after hyperparamer tuning. To examine which feature is important, I introduced a feature with random numbers. A feature can be considered as important If the importance of that feature is larger than that of the random feature. I subdivided the features that were found to be important as follows:

- Features related to the driver: education level, marital status, annual income, age, satefy rating, living address, whether the driver have changed living address in past 1 year, the number of claims the driver reported in past 5 years
- Features related to the accident: whether the accident occurred in a parking lot, whether a witness is present, the estimated claim payout
- Features related to the vehicle: the price of the vehicle

For `zip_code`, target encoding resulted in a worse performance than a simple transformation into `latitude` and `longitude`. Although the class imbalance was not very serious in this dataset, I tried SMOTE to synthesize new examples for the minority class. It seemed that this only worsened the performance.

I deployed XGBoost model as a Flask web app, which can return the probability that an insurance claim is fraudulent based on user inputs.

<img src="https://i.imgur.com/QffJHZY.gif" width=500>
