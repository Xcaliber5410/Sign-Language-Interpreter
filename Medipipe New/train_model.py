import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data1 = pd.read_csv("help_fixed.csv", header=None)
data2 = pd.read_csv("attack_fixed.csv", header=None)

# data3 = pd.read_csv("datasetnew.csv", header=None)

data = pd.concat([data1, data2], ignore_index=True)

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert features to numeric safely
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows with invalid values
X = X.dropna()
y = y.loc[X.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")