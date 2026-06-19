import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Dataset
df = pd.read_csv("customer_purchase_dataset.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features and Target
X = df.drop(["Customer_ID", "Purchased"], axis=1).values
y = df["Purchased"].values

# Enter K value
k = int(input("Enter K value: "))

# Shuffle Data
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Create Folds Manually
fold_size = len(X) // k

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for fold in range(k):

    start = fold * fold_size

    if fold == k - 1:
        end = len(X)
    else:
        end = start + fold_size

    # Test Data
    X_test = X[start:end]
    y_test = y[start:end]

    # Train Data
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    # Train Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print("\nFold", fold + 1)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

# Average Performance
print("\n========== Average Performance ==========")
print("Average Accuracy :", round(sum(accuracy_list)/k, 4))
print("Average Precision:", round(sum(precision_list)/k, 4))
print("Average Recall   :", round(sum(recall_list)/k, 4))
print("Average F1 Score :", round(sum(f1_list)/k, 4))
