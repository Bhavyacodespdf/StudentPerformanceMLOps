import joblib
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/Student Performance Factors.csv"
TARGET = "Exam_Score"

def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y
  
def get_preprocessor(X):
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor

def train_and_log(model, model_name, X_train, X_test, y_train, y_test, preprocessor):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model", model_name)

        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        mlflow.sklearn.log_model(
          pipeline,
          name="model",
          serialization_format="cloudpickle"
        )
        
    print(f"\nModel: {model_name}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return pipeline, r2
  
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)

    args = parser.parse_args()

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train.select_dtypes(include="number").agg(["mean", "std"]).to_json(
    "model/training_stats.json"
    )

    preprocessor = get_preprocessor(X_train)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Student Performance Prediction")

    experiments = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest 100", RandomForestRegressor(
            n_estimators=args.n_estimators,
            random_state=42
        )),
        ("Random Forest 200", RandomForestRegressor(
            n_estimators=200,
            max_depth=args.max_depth,
            random_state=42
        ))
    ]

    best_model = None
    best_r2 = float("-inf")
    
    for name, model in experiments:
      pipeline, r2 = train_and_log(
        model, name,
        X_train, X_test,
        y_train, y_test,
        preprocessor
      )
      
      if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
      
    joblib.dump(best_model, "model/model.joblib")
    print("\nBest model saved successfully!")


if __name__ == "__main__":
    main()