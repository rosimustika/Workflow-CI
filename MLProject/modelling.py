import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PATH DATA
# =========================
DATA_DIR = "loan_approval_preprocessing"

# =========================
# PATH ARTEFAK (UNTUK SKILLED)
# =========================
ARTIFACT_DIR = "../artifacts"


def main():
    # Autolog (tetap dipakai, tidak masalah untuk CI)
    mlflow.autolog()

    # Load data
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        # =========================
        # SIMPAN ARTEFAK KE REPO
        # =========================
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        artifact_path = os.path.join(ARTIFACT_DIR, "model.joblib")
        joblib.dump(model, artifact_path)

        print("Training selesai")
        print("Accuracy:", acc)
        print("Model disimpan di:", artifact_path)


if __name__ == "__main__":
    main()
