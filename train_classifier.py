import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# LOAD EMBEDDINGS
# ----------------------------
X = np.load("embeddings/X.npy")
y = np.load("embeddings/y.npy")

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ----------------------------
# SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\n🚀 Training Random Forest...")
model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, "models/emotion_rf_model.pkl")

print("\n✅ Model saved to models/emotion_rf_model.pkl")