import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Load Dataset
df = pd.read_csv("student_performance_dataset.csv")

# Features and Target
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

# User chooses K value
k = int(input("Enter the number of folds (K): "))

# K-Fold Cross Validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

r2_scores = []
mae_scores = []
mse_scores = []
rmse_scores = []

fold = 1

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2_scores.append(r2)
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)

    print(f"\nFold {fold}")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    fold += 1

print("\n===== Average Results =====")
print(f"Average R² Score : {np.mean(r2_scores):.4f}")
print(f"Average MAE      : {np.mean(mae_scores):.4f}")
print(f"Average MSE      : {np.mean(mse_scores):.4f}")
print(f"Average RMSE     : {np.mean(rmse_scores):.4f}")
