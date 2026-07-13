import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("churn_dataset.csv")

# -----------------------
# Preprocessing
# -----------------------
df = pd.read_csv("churn_dataset.csv")

# Remove customerID
df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------
# Feature Scaling
# -----------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# Balance Classes
# -----------------------
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# -----------------------
# Train Model
# -----------------------
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
predictions = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

# -----------------------
# Save Model
# -----------------------
joblib.dump(xgb_model, "churn_model.pkl")
joblib.dump(feature_names, "feature_names.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as churn_model.pkl")
print("Scaler saved as scaler.pkl")