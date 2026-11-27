import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

# 2. Remove unwanted columns
df = df.drop(columns=[
    "Unnamed: 0",
    "ID",
    "copd",
    "HAD",
    "IHD"
])

# 3. Separate input features and target
X = df.drop(columns=["COPDSEVERITY"])
y = df["COPDSEVERITY"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Fill missing values using training data median
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns
)

# 6. Create folder to store processed data
os.makedirs("data/processed", exist_ok=True)

# 7. Save processed training and testing data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)


# 8. Display results
print("Data preprocessing completed!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Missing values in training data:", X_train.isnull().sum().sum())
print("Missing values in testing data:", X_test.isnull().sum().sum())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())