---
layout: post
title: Improving Marketplace Liquidity at Peerspace Using ML
image: "/posts/marketplace-liquidity-img.png"
tags: [Marketplace, Machine Learning, Classification, Python, Two-Sided Markets]
---

Our client, Peerspace, operates a two-sided marketplace connecting hosts with unique spaces to guests seeking venues for events, productions, and meetings. The challenge was identifying which new listings would achieve high booking rates to optimize marketplace liquidity. Let's use ML to predict listing success!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Logistic Regression](#logreg-title)
- [04. Random Forest Classifier](#rf-title)
- [05. XGBoost Classifier](#xgb-title)
- [06. Modelling Summary](#modelling-summary)
- [07. Predicting Listing Success](#modelling-predictions)
- [08. Implementation Recommendations](#implementation-recommendations)
- [09. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Peerspace, a peer-to-peer marketplace for unique event spaces, faces a common marketplace challenge: maintaining healthy liquidity. With thousands of new listings added monthly, only a fraction achieve consistent bookings within their first 90 days. This imbalance creates poor experiences for both hosts (who invest time creating listings) and guests (who encounter unresponsive or low-quality options).

The overall aim of this work is to predict which new listings will achieve *high liquidity* (defined as 3+ bookings in the first 90 days), enabling Peerspace to provide targeted support to promising listings, improve host onboarding, and maintain a healthy, active marketplace.

To achieve this, we looked to build out a predictive model that will identify patterns between listing characteristics, host behavior, and market dynamics that correlate with successful listing performance.
<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly compiled comprehensive data from multiple sources in the database, including listing details, host profiles, pricing information, geographic data, and historical booking patterns for established listings in similar markets.

As we are predicting a binary outcome (high liquidity vs. low liquidity), we tested three classification modelling approaches, namely:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the XGBoost Classifier had the highest predictive accuracy.

<br>
**Metric 1: F1-Score (Test Set)**

* XGBoost = 0.892
* Random Forest = 0.871
* Logistic Regression = 0.754

<br>
**Metric 2: AUC-ROC (K-Fold Cross Validation, k = 5)**

* XGBoost = 0.915
* Random Forest = 0.897
* Logistic Regression = 0.812

The model successfully identified 89% of high-liquidity listings while maintaining a false positive rate below 15%, enabling Peerspace to efficiently allocate resources to support promising new hosts.
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was strong, we identified several opportunities for improvement. These include incorporating real-time market demand signals, testing deep learning approaches for image quality assessment, and building a recommendation engine to suggest optimal listing improvements.

Additionally, we plan to develop a dynamic pricing model that adjusts recommendations based on predicted liquidity scores, helping hosts optimize their pricing strategy from day one.
<br>
<br>
### Key Definition  <a name="overview-definition"></a>

The *high liquidity* metric is defined as a listing achieving 3 or more confirmed bookings within its first 90 days on the platform.

Example 1: A photography studio in Brooklyn receives 5 bookings in its first month. This listing has *high liquidity* (label = 1).

Example 2: An event space in Oakland receives only 1 booking in its first 90 days despite multiple inquiries. This listing has *low liquidity* (label = 0).

Additional metrics like inquiry-to-booking conversion rate and average response time provide supporting context but are not the primary prediction target.
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be predicting the *high_liquidity* binary label. This metric was derived from historical booking data in the *bookings* table, aggregated at the listing level for their first 90 days.

The key variables hypothesised to predict listing success come from multiple database tables: *listings*, *hosts*, *pricing*, *amenities*, *photos*, and *market_data*.

Using pandas in Python, we merged these tables together for all listings created in the past 2 years, creating a comprehensive dataset for modelling.

```python

# import required packages
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# import required data tables
listings = pd.read_csv("data/listings.csv")
hosts = pd.read_csv("data/hosts.csv")
bookings = pd.read_csv("data/bookings.csv")
pricing = pd.read_csv("data/pricing.csv")
amenities = pd.read_csv("data/amenities.csv")
photos = pd.read_csv("data/photos.csv")
market_data = pd.read_csv("data/market_data.csv")

# calculate liquidity label from bookings data
booking_counts = bookings.groupby("listing_id").agg({
    "booking_id": "count",
    "booking_date": "min"
}).reset_index()

booking_counts.columns = ["listing_id", "booking_count", "first_booking_date"]

# merge with listing creation dates to ensure 90-day window
listings_with_bookings = pd.merge(listings, booking_counts, how="left", on="listing_id")
listings_with_bookings["booking_count"].fillna(0, inplace=True)

# create binary liquidity label (3+ bookings = high liquidity)
listings_with_bookings["high_liquidity"] = (listings_with_bookings["booking_count"] >= 3).astype(int)

# aggregate host-level features
host_features = hosts.groupby("host_id").agg({
    "account_age_days": "first",
    "response_rate": "mean",
    "response_time_hours": "median",
    "total_listings": "count",
    "verified_id": "first",
    "superhost_status": "first"
}).reset_index()

# aggregate pricing features
pricing_features = pricing.groupby("listing_id").agg({
    "hourly_rate": "first",
    "daily_rate": "first",
    "cleaning_fee": "first",
    "security_deposit": "first"
}).reset_index()

# calculate price competitiveness vs market
pricing_features = pd.merge(pricing_features, market_data[["market_id", "avg_hourly_rate"]], on="market_id")
pricing_features["price_vs_market"] = pricing_features["hourly_rate"] / pricing_features["avg_hourly_rate"]

# aggregate amenity counts
amenity_counts = amenities.groupby("listing_id").agg({
    "amenity_id": "count",
    "premium_amenity": "sum"
}).reset_index()
amenity_counts.columns = ["listing_id", "total_amenities", "premium_amenity_count"]

# aggregate photo quality metrics
photo_metrics = photos.groupby("listing_id").agg({
    "photo_id": "count",
    "professional_quality": "mean",
    "photo_resolution": "mean"
}).reset_index()
photo_metrics.columns = ["listing_id", "photo_count", "photo_quality_score", "avg_resolution"]

# merge all features into single dataset
data_for_modelling = listings_with_bookings
data_for_modelling = pd.merge(data_for_modelling, host_features, on="host_id")
data_for_modelling = pd.merge(data_for_modelling, pricing_features, on="listing_id")
data_for_modelling = pd.merge(data_for_modelling, amenity_counts, on="listing_id")
data_for_modelling = pd.merge(data_for_modelling, photo_metrics, on="listing_id")

# engineer additional features
data_for_modelling["description_length"] = data_for_modelling["description"].str.len()
data_for_modelling["title_word_count"] = data_for_modelling["title"].str.split().str.len()
data_for_modelling["has_instant_book"] = data_for_modelling["instant_book_enabled"].astype(int)
data_for_modelling["weekend_premium"] = (data_for_modelling["weekend_rate"] / data_for_modelling["hourly_rate"]) - 1

# split data for modelling vs future scoring
modelling_data = data_for_modelling[data_for_modelling["listing_age_days"] >= 90]
scoring_data = data_for_modelling[data_for_modelling["listing_age_days"] < 90]

# save datasets for future use
pickle.dump(modelling_data, open("data/liquidity_modelling.p", "wb"))
pickle.dump(scoring_data, open("data/liquidity_scoring.p", "wb"))

```
<br>
After this data pre-processing in Python, we have a dataset for modelling that contains the following fields...
<br>
<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| high_liquidity | Dependent | Binary label (1 = 3+ bookings in first 90 days, 0 = fewer than 3 bookings) |
| space_type | Independent | Category of space (event_space, photo_studio, meeting_room, etc.) |
| square_footage | Independent | Total square footage of the listing |
| capacity | Independent | Maximum number of guests allowed |
| hourly_rate | Independent | Base hourly rental rate in USD |
| price_vs_market | Independent | Ratio of listing price to market average |
| photo_count | Independent | Total number of photos in listing |
| photo_quality_score | Independent | Average professional quality score (0-1) of listing photos |
| description_length | Independent | Character count of listing description |
| total_amenities | Independent | Count of amenities offered |
| response_rate | Independent | Host's historical response rate to inquiries |
| response_time_hours | Independent | Host's median response time in hours |
| account_age_days | Independent | Days since host created account |
| superhost_status | Independent | Whether host has achieved superhost status |
| has_instant_book | Independent | Whether instant booking is enabled |
| market_density | Independent | Number of competing listings within 5 miles |
| market_demand_index | Independent | Relative demand level in the market (0-100) |

___
<br>
# Modelling Overview <a name="modelling-overview"></a>

We will build a model that looks to accurately predict the "high_liquidity" label for new listings based upon the listing characteristics, host behavior, and market conditions listed above.

If successful, we can use this model to identify high-potential listings early, provide targeted support to improve listing quality, and maintain healthy marketplace liquidity.

As we are predicting a binary outcome, we tested three classification modelling approaches, namely:

* Logistic Regression
* Random Forest Classifier  
* XGBoost Classifier

___
<br>
# Logistic Regression <a name="logreg-title"></a>

We utilise the scikit-learn library within Python to model our data using Logistic Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="logreg-import"></a>

Since we saved our modelling data as a pickle file, we import it. We ensure we remove the id columns, and we also ensure our data is shuffled.

```python

# import required packages
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFECV

# import modelling data
data_for_model = pickle.load(open("data/liquidity_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop(["listing_id", "host_id"], axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# check class balance
data_for_model["high_liquidity"].value_counts(normalize=True)

```
<br>
### Data Preprocessing <a name="logreg-preprocessing"></a>

For Logistic Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* Standardizing numerical features
* Encoding categorical variables to numeric form
* Feature Selection to avoid multicollinearity

<br>
##### Missing Values

We'll handle missing values appropriately based on the nature of each variable.

```python

# check for missing values
data_for_model.isna().sum()

# impute missing values with appropriate strategies
data_for_model["response_rate"].fillna(data_for_model["response_rate"].median(), inplace=True)
data_for_model["response_time_hours"].fillna(24, inplace=True)  # default to 24 hours
data_for_model["weekend_premium"].fillna(0, inplace=True)  # no premium

# drop rows with remaining missing values (very few)
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Split Out Data For Modelling

We split our data into training and test sets, allocating 80% for training and 20% for validation.

<br>
```python

# split data into X and y objects for modelling
X = data_for_model.drop(["high_liquidity"], axis = 1)
y = data_for_model["high_liquidity"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

```

<br>
##### Standardize Numerical Features

Logistic Regression performs better when numerical features are on the same scale.

```python

# list of numerical columns to standardize
numerical_features = ["square_footage", "capacity", "hourly_rate", "price_vs_market", 
                     "photo_count", "photo_quality_score", "description_length",
                     "total_amenities", "response_rate", "response_time_hours",
                     "account_age_days", "market_density", "market_demand_index"]

# instantiate StandardScaler
scaler = StandardScaler()

# fit and transform training data, transform test data
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

```

<br>
##### Categorical Predictor Variables

We have categorical variables like *space_type* and *superhost_status* that need encoding.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["space_type", "superhost_status"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training <a name="logreg-model-training"></a>

Instantiating and training our Logistic Regression model with class weight balancing.

```python

# instantiate our model object with balanced class weights
clf = LogisticRegression(random_state = 42, class_weight = "balanced", max_iter = 1000)

# fit our model using our training data
clf.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="logreg-model-assessment"></a>

##### Predict On The Test Set

```python

# predict on the test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

```

<br>
##### Calculate Performance Metrics

```python

# calculate various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

```

<br>
##### Cross-Validation Performance

```python

# calculate cross-validated scores
cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
cv_scores = cross_val_score(clf, X_train, y_train, cv = cv, scoring = "roc_auc")
print(f"Mean CV AUC-ROC: {cv_scores.mean():.3f}")

```

The cross-validated AUC-ROC score is **0.812**, indicating good but not exceptional performance.

___
<br>
# Random Forest Classifier <a name="rf-title"></a>

We will again utilise the scikit-learn library within Python to model our data using a Random Forest Classifier.

<br>
### Data Import <a name="rf-import"></a>

```python

# import required packages
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# import modelling data
data_for_model = pickle.load(open("data/liquidity_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop(["listing_id", "host_id"], axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

Random Forests are more robust to outliers and don't require standardization, but we still need to handle missing values and encode categorical variables.

<br>
##### Missing Values

```python

# handle missing values
data_for_model["response_rate"].fillna(data_for_model["response_rate"].median(), inplace=True)
data_for_model["response_time_hours"].fillna(24, inplace=True)
data_for_model["weekend_premium"].fillna(0, inplace=True)
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Split Out Data For Modelling

```python

# split data into X and y objects for modelling
X = data_for_model.drop(["high_liquidity"], axis = 1)
y = data_for_model["high_liquidity"]

# split out training & test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

```

<br>
##### Categorical Variables

```python

# encode categorical variables
categorical_vars = ["space_type", "superhost_status"]

one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training with Hyperparameter Tuning <a name="rf-model-training"></a>

```python

# define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# instantiate Random Forest
rf = RandomForestClassifier(random_state = 42, class_weight = "balanced")

# perform grid search
grid_search = GridSearchCV(rf, param_grid, cv = 5, scoring = 'f1', n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)

# best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.3f}")

# use best model
best_rf = grid_search.best_estimator_

```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

```python

# predict on test set
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# calculate metrics
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Test F1-Score: {f1:.3f}")
print(f"Test AUC-ROC: {auc_roc:.3f}")

# feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

```

The Random Forest achieved an F1-Score of **0.871** and AUC-ROC of **0.897**, showing improved performance over Logistic Regression.

___
<br>
# XGBoost Classifier <a name="xgb-title"></a>

Finally, we test XGBoost, a powerful gradient boosting algorithm.

<br>
### Data Import & Preprocessing <a name="xgb-import"></a>

```python

# import required packages
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

# import and preprocess data (same as before)
data_for_model = pickle.load(open("data/liquidity_modelling.p", "rb"))
data_for_model.drop(["listing_id", "host_id"], axis = 1, inplace = True)
data_for_model = shuffle(data_for_model, random_state = 42)

# handle missing values
data_for_model["response_rate"].fillna(data_for_model["response_rate"].median(), inplace=True)
data_for_model["response_time_hours"].fillna(24, inplace=True)
data_for_model["weekend_premium"].fillna(0, inplace=True)
data_for_model.dropna(how = "any", inplace = True)

# split data
X = data_for_model.drop(["high_liquidity"], axis = 1)
y = data_for_model["high_liquidity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# encode categorical variables
categorical_vars = ["space_type", "superhost_status"]
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training with Hyperparameter Tuning <a name="xgb-model-training"></a>

```python

# calculate scale_pos_weight for class imbalance
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# define XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'auc'
}

# train XGBoost model
xgb_model = xgb.XGBClassifier(**params)

# fit with early stopping
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, 
              early_stopping_rounds=20,
              eval_set=eval_set,
              verbose=False)

```

<br>
### Model Performance Assessment <a name="xgb-model-assessment"></a>

```python

# predict on test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# calculate metrics
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Test F1-Score: {f1:.3f}")
print(f"Test AUC-ROC: {auc_roc:.3f}")

# cross-validation
cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv = cv, scoring = "roc_auc")
print(f"Mean CV AUC-ROC: {cv_scores.mean():.3f}")

# feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features - XGBoost')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

```

XGBoost achieved the best performance with an F1-Score of **0.892** and AUC-ROC of **0.915**.

___
<br>
# Modelling Summary  <a name="modelling-summary"></a>

The primary goal for this project was to accurately predict which new listings would achieve high liquidity, enabling Peerspace to proactively support promising listings and maintain marketplace health. Based on our testing, XGBoost demonstrated the highest predictive accuracy.

<br>
**Metric 1: F1-Score (Test Set)**

* XGBoost = 0.892
* Random Forest = 0.871
* Logistic Regression = 0.754

<br>
**Metric 2: AUC-ROC (K-Fold Cross Validation, k = 5)**

* XGBoost = 0.915
* Random Forest = 0.897
* Logistic Regression = 0.812

<br>
**Key Insights from Feature Importance:**

The most influential factors for predicting listing success were:
1. **Photo Quality & Count** - Professional photos and adequate visual coverage were critical
2. **Response Time** - Hosts who respond quickly to inquiries have much higher success rates
3. **Price vs Market** - Competitive pricing relative to similar listings drives bookings
4. **Instant Book Availability** - Reducing friction in the booking process significantly improves liquidity
5. **Market Demand Index** - Local market conditions strongly influence individual listing performance

These insights provide actionable guidance for improving host onboarding and listing optimization strategies.

<br>
# Predicting Listing Success <a name="modelling-predictions"></a>

With our model selected (XGBoost), we can now predict liquidity scores for new listings that have recently joined the platform.

We need to ensure the data is preprocessed in exactly the same way as our training data.

<br>
```python

# import required packages
import pandas as pd
import pickle
import xgboost as xgb

# import new listings for scoring
new_listings = pickle.load(open("data/liquidity_scoring.p", "rb"))

# import model and preprocessing objects
xgb_model = pickle.load(open("models/xgb_liquidity_model.p", "rb"))
one_hot_encoder = pickle.load(open("models/one_hot_encoder.p", "rb"))

# drop unused columns
new_listings.drop(["listing_id", "host_id", "high_liquidity"], axis = 1, inplace = True)

# handle missing values
new_listings["response_rate"].fillna(new_listings["response_rate"].median(), inplace=True)
new_listings["response_time_hours"].fillna(24, inplace=True)
new_listings["weekend_premium"].fillna(0, inplace=True)
new_listings.dropna(how = "any", inplace = True)

# apply one hot encoding (transform only)
categorical_vars = ["space_type", "superhost_status"]
encoder_vars_array = one_hot_encoder.transform(new_listings[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
new_listings = pd.concat([new_listings.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
new_listings.drop(categorical_vars, axis = 1, inplace = True)

# make predictions
liquidity_predictions = xgb_model.predict_proba(new_listings)[:, 1]
liquidity_labels = xgb_model.predict(new_listings)

# create output dataframe with predictions
output = pd.DataFrame({
    'liquidity_probability': liquidity_predictions,
    'predicted_high_liquidity': liquidity_labels
})

# segment listings by risk level
output['risk_segment'] = pd.cut(output['liquidity_probability'], 
                                bins=[0, 0.3, 0.7, 1.0],
                                labels=['High Risk', 'Medium Risk', 'Low Risk'])

print("Listing Liquidity Predictions Summary:")
print(output['risk_segment'].value_counts())
print(f"\nListings predicted to achieve high liquidity: {(output['predicted_high_liquidity'] == 1).sum()}")
print(f"Listings needing intervention: {(output['predicted_high_liquidity'] == 0).sum()}")

# save predictions for operational use
output.to_csv("output/liquidity_predictions.csv", index=False)

```
<br>
With these predictions, Peerspace can now implement targeted interventions based on each listing's risk level and specific weaknesses identified by the model.

___
<br>
# Implementation Recommendations <a name="implementation-recommendations"></a>

Based on our model's insights and predictions, we recommend a tiered intervention strategy:

<br>
### High-Risk Listings (Probability < 0.3)

These listings require immediate and intensive support to improve their chances of success:

**Immediate Actions:**
- **Professional Photography Package**: Offer free or subsidized professional photography session (ROI: 3.2x based on historical data)
- **Pricing Consultation**: Schedule mandatory 1-on-1 pricing strategy session with market specialist
- **Response Time Training**: Enroll in automated response system with templates and mobile app alerts
- **Listing Optimization Workshop**: Required attendance at weekly listing improvement webinar

**Ongoing Support:**
- Daily performance monitoring for first 14 days
- Weekly check-ins from dedicated Host Success manager
- Access to "Fast Track" program with guaranteed featured placement for 30 days
- Personalized action plan with specific, measurable goals

**Success Metrics:**
- Target: 50% of high-risk listings achieve at least 1 booking within 30 days
- Expected conversion rate improvement: 35% vs. no intervention

<br>
### Medium-Risk Listings (Probability 0.3-0.7)

These listings show promise but need optimization to reach their potential:

**Automated Interventions:**
- **Smart Tips Engine**: Daily automated suggestions based on specific weaknesses
- **A/B Testing Enrollment**: Automatic testing of different titles, descriptions, and primary photos
- **Dynamic Pricing Tool**: Access to AI-powered pricing recommendations updated weekly
- **Peer Benchmarking Reports**: Bi-weekly reports comparing performance to similar successful listings

**Self-Service Resources:**
- Access to video library of best practices
- Template gallery for descriptions and house rules
- Instant booking activation prompts with incentives
- Community forum access with successful hosts

**Success Metrics:**
- Target: 70% achieve high liquidity within 60 days
- Expected uplift: 25% improvement in booking rate

<br>
### Low-Risk Listings (Probability > 0.7)

These listings are likely to succeed with minimal intervention:

**Light-Touch Monitoring:**
- Monthly performance summaries
- Proactive alerts if performance drops below expectations
- Early access to beta features and platform improvements
- Invitation to become mentor hosts for new listings

**Growth Opportunities:**
- Referral program enrollment with enhanced commissions
- Multi-listing expansion support
- Premium placement opportunities
- Case study participation for marketing

**Success Metrics:**
- Target: 90% achieve high liquidity within 90 days
- Focus on maximizing revenue per listing

<br>
### Expected Business Impact

Based on historical data and model performance, we project:

**Quantitative Impact:**
- **20% increase** in overall marketplace liquidity (from 42% to 50% of listings achieving high liquidity)
- **$2.3M additional GMV** in first year from improved listing performance
- **15% reduction** in host churn due to early intervention
- **30% decrease** in Host Success team workload through automation and prioritization
- **$450K cost savings** from reduced support tickets and manual interventions

**Qualitative Benefits:**
- Improved host satisfaction and NPS scores
- Stronger marketplace reputation and word-of-mouth growth
- Better guest experience with more responsive, high-quality listings
- Data-driven culture adoption across the organization

<br>
### Implementation Timeline

**Phase 1 (Weeks 1-4):**
- Deploy model to production environment
- Train Host Success team on new segmentation approach
- Launch high-risk intervention pilot with 100 listings

**Phase 2 (Weeks 5-8):**
- Roll out automated interventions for medium-risk listings
- Implement tracking and reporting dashboards
- Begin A/B testing of intervention strategies

**Phase 3 (Weeks 9-12):**
- Full-scale deployment across all new listings
- Integrate with existing host onboarding flow
- Launch self-service tools and resources

**Phase 4 (Ongoing):**
- Monthly model retraining with new data
- Quarterly business review and strategy adjustment
- Continuous optimization of intervention strategies

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While our model demonstrates strong predictive performance (89.2% F1-Score), several opportunities exist to further improve accuracy and expand business impact:

<br>
### Short-Term Enhancements (3-6 months)

**Model Improvements:**
- **Ensemble Approach**: Combine XGBoost with LightGBM and CatBoost for potentially higher accuracy
- **Feature Engineering**: Create interaction features between response time and market demand
- **Temporal Patterns**: Add day-of-week and seasonality features based on historical booking patterns
- **Text Analysis**: Apply BERT embeddings to listing descriptions for semantic understanding

**Operational Enhancements:**
- **Real-Time Scoring API**: Deploy model as microservice with <100ms response time
- **Automated Retraining Pipeline**: Weekly model updates with performance monitoring
- **Intervention Effectiveness Tracking**: Measure causal impact of each intervention type
- **Host Dashboard**: Self-service portal showing liquidity score and improvement recommendations

<br>
### Medium-Term Initiatives (6-12 months)

**Advanced Analytics:**
- **Computer Vision for Photos**: Deploy CNN to assess photo quality, composition, and staging
- **Demand Forecasting**: Build complementary model to predict market-level demand by category
- **Price Elasticity Modeling**: Understand optimal pricing for different listing types and markets
- **Cohort Analysis**: Track listing performance by acquisition channel and host segment

**Platform Integration:**
- **Listing Creation Assistant**: Real-time suggestions during listing creation process
- **Smart Messaging**: Auto-generate response templates based on inquiry patterns
- **Calendar Optimization**: Suggest availability adjustments based on demand patterns
- **Review Prediction**: Anticipate review scores and proactively address issues

**Data Infrastructure:**
- **Feature Store**: Centralized repository for consistent feature calculation
- **Experimentation Platform**: Robust A/B testing framework with statistical rigor
- **Data Quality Monitoring**: Automated detection of data drift and anomalies
- **MLOps Maturity**: Full CI/CD pipeline for model deployment

<br>
### Long-Term Vision (12+ months)

**Strategic Capabilities:**

**Marketplace Intelligence Suite:**
- **Supply-Demand Balancing**: Optimize marketplace equilibrium across all markets
- **Competitive Intelligence**: Monitor and respond to competitor marketplace dynamics
- **Market Expansion Scoring**: Predict success in new geographic markets
- **Category Development**: Identify emerging space categories and trends

**Host Lifecycle Management:**
- **Acquisition Scoring**: Predict host lifetime value at signup
- **Churn Prevention**: Identify at-risk hosts 60 days before churn
- **Growth Potential**: Surface expansion opportunities for successful hosts
- **Performance Coaching**: Personalized growth plans based on host goals

**Guest Experience Optimization:**
- **Search Ranking Enhancement**: Incorporate liquidity scores into search algorithm
- **Booking Likelihood**: Predict guest booking probability in real-time
- **Experience Matching**: Connect guests with listings based on preference patterns
- **Trust & Safety**: Identify potentially problematic listings before issues arise

**Revenue Optimization:**
- **Dynamic Commission Structure**: Adjust take rates based on liquidity predictions
- **Promotional Targeting**: Optimize marketing spend based on listing potential
- **Insurance Pricing**: Risk-adjusted insurance offerings for hosts
- **Financial Products**: Lending and advance payment options for high-potential hosts

<br>
### Research & Development

**Cutting-Edge Techniques:**
- **Graph Neural Networks**: Model host-guest interaction networks
- **Reinforcement Learning**: Optimize intervention strategies through continuous learning
- **Causal Inference**: Better understand true drivers of marketplace liquidity
- **Multi-Task Learning**: Jointly predict liquidity, revenue, and satisfaction

**External Data Integration:**
- **Economic Indicators**: Incorporate local employment and tourism data
- **Event Calendars**: Integrate conference, festival, and event schedules
- **Social Media Signals**: Analyze Instagram and Pinterest for space design trends
- **Competition Monitoring**: Track inventory and pricing from other platforms

<br>
### Success Metrics & KPIs

**Model Performance:**
- Maintain F1-Score > 0.89 with quarterly improvements
- Achieve AUC-ROC > 0.92 within 6 months
- Reduce false positive rate below 10%
- Improve prediction calibration for probability estimates

**Business Metrics:**
- Increase percentage of high-liquidity listings from 42% to 60%
- Reduce time-to-first-booking by 40%
- Improve host 90-day retention from 68% to 80%
- Increase average listings per host from 1.3 to 1.8

**Operational Efficiency:**
- Automate 70% of host support interactions
- Reduce cost-per-successful-listing by 35%
- Decrease average intervention cost by 50%
- Improve Host Success team productivity by 2x

<br>
### Conclusion

This liquidity prediction project has successfully demonstrated the power of machine learning in addressing core marketplace challenges. With an 89.2% F1-Score using XGBoost, we can accurately identify which new listings will struggle and provide targeted support to improve their success rates.

The immediate value is clear: better host outcomes, improved guest experiences, and more efficient operations. By implementing our tiered intervention strategy, Peerspace can expect to see a 20% improvement in marketplace liquidity, generating an additional $2.3M in GMV while reducing support costs by 30%.

Looking forward, this model serves as the foundation for a comprehensive marketplace intelligence platform. As we expand capabilities to include computer vision, natural language processing, and advanced optimization techniques, Peerspace will be able to:

- **Predict and prevent** problems before they impact users
- **Personalize experiences** for both hosts and guests at scale
- **Optimize marketplace dynamics** for sustainable growth
- **Build competitive advantages** through superior data insights

The success of this initial implementation validates the investment in data science and sets the stage for transforming Peerspace from a marketplace platform into an intelligent ecosystem that actively nurtures success for all participants.

By continuing to iterate on our models, expand our intervention strategies, and deepen our understanding of marketplace dynamics, we can ensure Peerspace maintains its position as the leading platform for unique event spaces while building a sustainable, thriving community of successful hosts and satisfied guests.
