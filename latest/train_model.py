import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Specify your list of files here
DATASET_PATHS = ["dataset_features.csv", "help.csv","attack.csv","house.csv","male.csv","numbers_dataset.csv"]

def train():
    all_data = []

    # 2. Iterate and load each file
    for path in DATASET_PATHS:
        if os.path.exists(path):
            print(f"Loading {path}...")
            df = pd.read_csv(path, header=None)
            all_data.append(df)
        else:
            print(f"Warning: {path} not found. Skipping...")

    if not all_data:
        print("Error: No data files were found. Training aborted.")
        return

    # 3. Combine all DataFrames into one
    data = pd.concat(all_data, ignore_index=True)

    # Drop any corrupted rows
    data = data.dropna()

    # Features (first 146 columns) and Labels (last column)
    X = data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    y = data.iloc[:, -1].astype(str)

    # Clean completely one more time
    data_clean = pd.concat([X, y], axis=1).dropna()
    X_clean = data_clean.iloc[:, :-1]
    y_clean = data_clean.iloc[:, -1]

    print(f"Training on {len(X_clean)} samples with {X_clean.shape[1]} features...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # Use n_jobs=-1 to train on all CPU cores for speed
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "gesture_model.pkl")
    print("Model successfully saved as gesture_model.pkl")

if __name__ == "__main__":
    train()