import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score


projects = pd.read_csv("train.csv")

y = np.log1p(projects["hours_spent"])

drop = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
X = projects.drop(columns=drop)

cat_cols = X.select_dtypes(include=["str", "object"]).columns
X[cat_cols] = X[cat_cols].astype("category")

# Replacing nan cus i think dividning by zero is causing that error
X['commits_per_day'] = X['num_commits'] / X['active_commit_days'].replace(0, np.nan)
X['total_changed'] = X['lines_added_total'] + X['lines_deleted_total']
X['active_day_ratio'] = X['active_commit_days'] / X['project_span_days'].replace(0, np.nan)
X['added_lines_per_day'] = X['lines_added_total'] / X['active_commit_days'].replace(0, np.nan)
X['removed_lines_per_day'] = X['lines_deleted_total'] / X['active_commit_days'].replace(0, np.nan)
X['changed_lines_per_day'] = X['total_changed'] / X['active_commit_days'].replace(0, np.nan)

# Text columns -> category columns
for col in cat_cols:
    X[col] = X[col].astype("category")

# X and y are done, next bit is the actual model

project_model = HistGradientBoostingRegressor(
    categorical_features="from_dtype", 
    learning_rate=0.03, max_iter=10000,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=100, random_state=42)

project_model.fit(X, y)

# scores = cross_val_score(project_model, X, y, cv=5,
#                         scoring="neg_root_mean_squared_error")
# print(-scores.mean(), scores.std())

test = pd.read_csv("test.csv")

testX = test.drop(columns=drop, errors="ignore")


testX['commits_per_day'] = testX['num_commits'] / testX['active_commit_days'].replace(0, np.nan)
testX['total_changed'] = testX['lines_added_total'] + testX['lines_deleted_total']
testX['active_day_ratio'] = testX['active_commit_days'] / testX['project_span_days'].replace(0, np.nan)
testX['added_lines_per_day'] = testX['lines_added_total'] / testX['active_commit_days'].replace(0, np.nan)
testX['removed_lines_per_day'] = testX['lines_deleted_total'] / testX['active_commit_days'].replace(0, np.nan)
testX['changed_lines_per_day'] = testX['total_changed'] / testX['active_commit_days'].replace(0, np.nan)

for col in X.select_dtypes(include="category").columns:
    testX[col] = pd.Categorical(testX[col], categories=X[col].cat.categories)

predictions = np.expm1(project_model.predict(testX)) # exponent cus the hours are predicted in their log form

# I think cus the hours are rounded, I have a better chance if i round it

# Since we dropped project_id i have to manually add it as the first column
pd.DataFrame({"project_id" : test["project_id"], "hours_spent": predictions}).to_csv("predictions.csv", index=False)
