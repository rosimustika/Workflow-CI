import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# PATH DATA
# =========================
DATA_DIR = "loan_approval_preprocessing"


def main():
    # =========================
    # LOAD DATA
    # =========================
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with mlflow.start_run():
        # =========================
        # TRAIN MODEL
        # =========================
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # =========================
        # METRICS (MANUAL)
        # =========================
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # =========================
        # PARAMS
        # =========================
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # =========================
        # ARTEFAK (WAJIB)
        # =========================

        # Artefak 1: MODEL (INI YANG DINILAI)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train[:1]
        )

        # Artefak 2: Script training
        mlflow.log_artifact(__file__)

        # Artefak 3: Sample prediction
        np.save("sample_prediction.npy", preds[:10])
        mlflow.log_artifact("sample_prediction.npy")

        print("✅ Training selesai")
        print("Accuracy:", acc)


if __name__ == "__main__":
    main()
