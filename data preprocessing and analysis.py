import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET

df = pd.read_csv("dataset1.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)

# IDENTIFY NUMERICAL & CATEGORICAL COLUMNS

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# 1. MISSING VALUE ANALYSIS

print("\nMISSING VALUE ANALYSIS")

missing_count = df.isnull().sum().sum()

if missing_count > 0:

    print(f"Missing Values Found: {missing_count}")

    # Numerical Columns → Median Imputation
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical Columns → Mode Imputation
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values handled.")

else:
    print("No Missing Values Found.")

# 2. DUPLICATE RECORD ANALYSIS

print("\nDUPLICATE ANALYSIS")

duplicates = df.duplicated().sum()

if duplicates > 0:

    print(f"Duplicate Records Found: {duplicates}")

    df.drop_duplicates(inplace=True)

    print("Duplicates Removed.")

else:
    print("No Duplicate Records Found.")

# 3. OUTLIER DETECTION USING IQR

print("\nOUTLIER ANALYSIS")

outlier_found = False

for col in num_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    if outliers > 0:
        outlier_found = True
        print(f"{col}: {outliers} Outliers Found")

# OUTLIER TREATMENT USING IQR CAPPING

if outlier_found:

    print("\nApplying IQR Capping...")

    for col in num_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    print("Outliers Treated.")

else:
    print("No Significant Outliers Found.")

# BASIC STATISTICAL ANALYSIS

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


plt.figure(figsize=(8,5))

plt.hist(df['Total_Sales'],
         bins=15,
         edgecolor='black')

plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Number of Transactions')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# VISUALIZATION 2 - BOX PLOT

if len(num_cols) > 0:

    plt.figure(figsize=(7, 5))

    sns.boxplot(x=df[num_cols[0]])

    plt.title(f'Box Plot - {num_cols[0]}')

    plt.show()

# VISUALIZATION 3 - SCATTER PLOT

if len(num_cols) >= 2:

    plt.figure(figsize=(7, 5))

    plt.scatter(df[num_cols[0]], df[num_cols[1]])

    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])

    plt.title(f'{num_cols[0]} vs {num_cols[1]}')

    plt.show()

# VISUALIZATION 4 - CORRELATION HEATMAP

if len(num_cols) >= 2:

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df[num_cols].corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title('Correlation Heatmap')

    plt.show()

# VISUALIZATION 5 - BAR CHART

plt.figure(figsize=(8,5))

df['Product_Category'].value_counts().plot(
    kind='bar',
    edgecolor='black'
)

plt.title('Number of Transactions by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Count')

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
