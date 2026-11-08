
import json
import pandas as pd
import mlflow
from pathlib import Path

TRAINING_STATS = "model/training_stats.json"
PREDICTION_LOG = "prediction_logs.csv"
DRIFT_THRESHOLD = 0.20


def check_drift():
    with open(TRAINING_STATS, "r") as file:
        training_stats = json.load(file)

    if not Path(PREDICTION_LOG).exists():
        print("No prediction logs found.")
        return

    logs = pd.read_csv(PREDICTION_LOG)

    print("\nDATA DRIFT REPORT")
    print("=" * 40)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Student Performance Prediction")

    with mlflow.start_run(run_name="Drift Monitoring"):

        mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
        mlflow.log_param("prediction_records", len(logs))

        overall_drift = False

        for feature, stats in training_stats.items():

            if feature not in logs.columns:
                continue

            values = logs[feature].dropna()

            if len(values) < 10:
                print(f"\n{feature}: Insufficient data")
                continue

            current_mean = values.mean()
            current_std = values.std()

            mean_drift = (
                abs(current_mean - stats["mean"])
                > DRIFT_THRESHOLD * abs(stats["mean"])
            )

            std_drift = (
                abs(current_std - stats["std"])
                > DRIFT_THRESHOLD * abs(stats["std"])
            )

            drift = mean_drift or std_drift

            if drift:
                overall_drift = True

            print(f"\nFeature: {feature}")
            print(f"Training mean: {stats['mean']:.2f}")
            print(f"Current mean: {current_mean:.2f}")
            print(f"Training std: {stats['std']:.2f}")
            print(f"Current std: {current_std:.2f}")
            print(f"Mean drift: {'Yes' if mean_drift else 'No'}")
            print(f"Std drift: {'Yes' if std_drift else 'No'}")
            print("Status:", "DRIFT DETECTED" if drift else "Stable")

            mlflow.log_metrics({
                f"{feature}_current_mean": float(current_mean),
                f"{feature}_current_std": float(current_std),
                f"{feature}_mean_drift": int(mean_drift),
                f"{feature}_std_drift": int(std_drift)
            })

        mlflow.set_tag(
            "overall_drift_status",
            "DRIFT DETECTED" if overall_drift else "STABLE"
        )

        print("\nMLflow: Drift monitoring results logged successfully!")


if __name__ == "__main__":
    check_drift()
