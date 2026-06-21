# train.py
# Purpose: Hyperparameter tuning + MLflow tracking + model
#          registration on Hugging Face Model Hub (production)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ── MLflow tracking 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-pipeline")

# ── HF Configuration 
HF_USERNAME   = "nikhilnayakbv"
DATASET_REPO  = f"{HF_USERNAME}/tourism-wellness"
MODEL_REPO    = f"{HF_USERNAME}/tourism-wellness-model"
MODEL_FILE    = "best_tourism_model_v1.joblib"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Load train/test from Hugging Face 
print(" Loading train/test data from Hugging Face ...")
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").squeeze()
print(f"   Train: {Xtrain.shape} | Test: {Xtest.shape}")

# ── Feature definitions 
NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# ── Class weight for imbalance 
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ── Preprocessing + model pipeline 
preprocessor = make_column_transformer(
    (StandardScaler(), NUMERIC_FEATURES),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ── Full hyperparameter grid (production) 
param_grid = {
    "xgbclassifier__n_estimators":       [50, 75, 100, 125],
    "xgbclassifier__max_depth":          [2, 3, 4],
    "xgbclassifier__learning_rate":      [0.01, 0.05, 0.1],
    "xgbclassifier__colsample_bytree":   [0.5, 0.6],
    "xgbclassifier__colsample_bylevel":  [0.5, 0.6],
    "xgbclassifier__reg_lambda":         [0.4, 0.6],
}

CLASSIFICATION_THRESHOLD = 0.45

# ── MLflow run 
with mlflow.start_run(run_name="production_grid_search"):

    grid_search = GridSearchCV(
        model_pipeline, param_grid,
        cv=5, n_jobs=-1, scoring="f1"
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all combinations as nested runs
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(cv_results["params"][i])
            mlflow.log_metric("mean_test_f1", cv_results["mean_test_score"][i])
            mlflow.log_metric("std_test_f1",  cv_results["std_test_score"][i])

    # Parent run: best params + final metrics
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= CLASSIFICATION_THRESHOLD).astype(int)

    train_rep = classification_report(ytrain, y_pred_train, output_dict=True)
    test_rep  = classification_report(ytest,  y_pred_test,  output_dict=True)

    mlflow.log_metrics({
        "train_accuracy":  train_rep["accuracy"],
        "train_precision": train_rep["1"]["precision"],
        "train_recall":    train_rep["1"]["recall"],
        "train_f1":        train_rep["1"]["f1-score"],
        "test_accuracy":   test_rep["accuracy"],
        "test_precision":  test_rep["1"]["precision"],
        "test_recall":     test_rep["1"]["recall"],
        "test_f1":         test_rep["1"]["f1-score"],
    })

    print("\ Best Parameters:", grid_search.best_params_)
    print("\ Test Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # ── Save model locally 
    joblib.dump(best_model, MODEL_FILE)
    mlflow.log_artifact(MODEL_FILE, artifact_path="model")
    print(f"\n Model saved as: {MODEL_FILE}")

    # ── Register on Hugging Face Model Hub 
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f" Model repo '{MODEL_REPO}' already exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print(f" Model repo '{MODEL_REPO}' created.")

    api.upload_file(
        path_or_fileobj=MODEL_FILE,
        path_in_repo=MODEL_FILE,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f" Model registered at: https://huggingface.co/{MODEL_REPO}")
