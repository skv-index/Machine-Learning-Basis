import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("sales_prediction_dataset.csv")

# BASIC INFORMATION

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# MISSING VALUES

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\nNo Missing Values Found")
else:
    print("\nMissing Values Exist")

# DUPLICATE VALUES

print("\n========== DUPLICATES ==========")

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates Removed")

# DESCRIPTIVE STATISTICS

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# OUTLIER DETECTION USING IQR

print("\n========== OUTLIER DETECTION ==========")

for col in df.columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col} : {len(outliers)} outliers")

print("\n========== OUTLIER FIXING ==========")
for col in df.columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("Outliers capped successfully")

# CORRELATION HEATMAP

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Heatmap")
plt.show()

# HISTOGRAMS
df.hist(figsize=(10,8), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# BOXPLOTS

plt.figure(figsize=(12,6))

for i, col in enumerate(df.columns, 1):
    plt.subplot(2,3,i)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# STANDARDIZATION

print("\n========== STANDARDIZATION ==========")

X = df.drop("Sales", axis=1)
y = df["Sales"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nBefore Scaling")
print(X.head())

print("\nAfter Scaling")
print(X_scaled.head())

print("\nMeans After Scaling")
print(X_scaled.mean())

print("\nStandard Deviations After Scaling")
print(X_scaled.std())

print("\nPreprocessing Completed Successfully")

# ==========================================
# MULTIPLE LINEAR REGRESSION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Features and Target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nMULTIPLE LINEAR REGRESSION ")
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape :", X_test.shape)

# Create Model
lr = LinearRegression()

# Train Model
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)


# PERFORMANCE METRICS

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMULTIPLE LINEAR REGRESSION RESULTS")
print("-----------------------------------")
print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R²   :", round(r2, 4))


# ACTUAL VS PREDICTED


comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nFirst 10 Predictions")
print(comparison.head(10))

# ==========================================
# POLYNOMIAL REGRESSION
# ==========================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Create Polynomial Features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create Model
poly_model = LinearRegression()

# Train Model
poly_model.fit(X_train_poly, y_train)

# Predict
y_pred_poly = poly_model.predict(X_test_poly)

# ==========================================
# PERFORMANCE METRICS
# ==========================================

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPOLYNOMIAL REGRESSION RESULTS")
print("-----------------------------------")
print("MAE  :", round(mae_poly, 4))
print("MSE  :", round(mse_poly, 4))
print("RMSE :", round(rmse_poly, 4))
print("R²   :", round(r2_poly, 4))

# Actual vs Predicted
comparison_poly = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred_poly
})

print("\nFirst 10 Predictions")
print(comparison_poly.head(10))


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Multiple Linear Regression")

plt.show()

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_poly)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Polynomial Regression")

plt.show()

# ==========================================
# MODEL COMPARISON
# ==========================================

comparison_metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Linear Regression": [
        round(mae, 4),
        round(mse, 4),
        round(rmse, 4),
        round(r2, 4)
    ],
    "Polynomial Regression": [
        round(mae_poly, 4),
        round(mse_poly, 4),
        round(rmse_poly, 4),
        round(r2_poly, 4)
    ]
})

print("\nMODEL COMPARISON")
print(comparison_metrics)

# ==========================================
# DETERMINE BEST MODEL
# ==========================================

print("\nMODEL ANALYSIS")

if mae_poly < mae:
    print("✓ Polynomial Regression has lower MAE")
else:
    print("✓ Linear Regression has lower MAE")

if mse_poly < mse:
    print("✓ Polynomial Regression has lower MSE")
else:
    print("✓ Linear Regression has lower MSE")

if rmse_poly < rmse:
    print("✓ Polynomial Regression has lower RMSE")
else:
    print("✓ Linear Regression has lower RMSE")

if r2_poly > r2:
    print("✓ Polynomial Regression has higher R² Score")
else:
    print("✓ Linear Regression has higher R² Score")

# Final Decision
poly_wins = 0
linear_wins = 0

if mae_poly < mae:
    poly_wins += 1
else:
    linear_wins += 1

if mse_poly < mse:
    poly_wins += 1
else:
    linear_wins += 1

if rmse_poly < rmse:
    poly_wins += 1
else:
    linear_wins += 1

if r2_poly > r2:
    poly_wins += 1
else:
    linear_wins += 1

print("\nFINAL CONCLUSION")

if poly_wins > linear_wins:
    print("Polynomial Regression is the BETTER model for this dataset.")
elif linear_wins > poly_wins:
    print("Linear Regression is the BETTER model for this dataset.")
else:
    print("Both models perform similarly on this dataset.")

# ==========================================
# USER INPUT PREDICTION
# ==========================================

import pandas as pd

print("\nENTER VALUES FOR SALES PREDICTION")

advertising_budget = float(input("Enter Advertising Budget: "))
store_size = float(input("Enter Store Size: "))
customers = int(input("Enter Number of Customers: "))
discount_percentage = float(input("Enter Discount Percentage: "))

# Create DataFrame with proper column names
new_data = pd.DataFrame({
    "Advertising_Budget": [advertising_budget],
    "Store_Size": [store_size],
    "Customers": [customers],
    "Discount_Percentage": [discount_percentage]
})

# ==========================================
# LINEAR REGRESSION PREDICTION
# ==========================================

linear_prediction = lr.predict(new_data)

print("\n===== LINEAR REGRESSION =====")
print("Predicted Sales:", round(linear_prediction[0], 2))

# ==========================================
# POLYNOMIAL REGRESSION PREDICTION
# ==========================================

new_data_poly = poly.transform(new_data)

poly_prediction = poly_model.predict(new_data_poly)

print("\n===== POLYNOMIAL REGRESSION =====")
print("Predicted Sales:", round(poly_prediction[0], 2))
