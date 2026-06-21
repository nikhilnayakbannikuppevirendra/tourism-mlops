# ─────────────────────────────────────────────────────────────
# prep.py
# Purpose: Load, clean, split, and register train/test splits
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# ── Configuration ────────────────────────────────────────────
HF_USERNAME = "nikhilnayakbv"           # ← Replace with your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness"
DATASET_PATH = f"hf://datasets/{DATASET_REPO_ID}/tourism.csv"
RANDOM_STATE = 42

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Step 1: Load dataset from Hugging Face ───────────────────
print("📥 Loading dataset from Hugging Face ...")
df = pd.read_csv(DATASET_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# ── Step 2: Data Cleaning ─────────────────────────────────────
# 2a. Drop unnecessary columns
DROP_COLS = ["Unnamed: 0", "CustomerID"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
print(f"\n🗑️  Dropped columns: {DROP_COLS}")

# 2b. Fix known encoding/inconsistency issues in Gender column
df["Gender"] = df["Gender"].str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 2c. Standardise MaritalStatus – treat 'Unmarried' as 'Single'
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 2d. Strip whitespace from all string columns
str_cols = df.select_dtypes(include="object").columns
for col in str_cols:
    df[col] = df[col].str.strip()

# 2e. Missing value check (already clean, but guard with median/mode imputation)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\n✅ Missing values after cleaning: {df.isnull().sum().sum()}")
print(f"   Final shape: {df.shape}")
print(f"\n📊 Target distribution:")
print(df["ProdTaken"].value_counts())

# ── Step 3: Define Features ───────────────────────────────────
TARGET = "ProdTaken"

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

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ── Step 4: Train/Test Split (stratified) ─────────────────────
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\n✂️  Split sizes → Train: {Xtrain.shape}, Test: {Xtest.shape}")

# ── Step 5: Save splits locally ───────────────────────────────
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv",  index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv",  index=False)
print("\n💾 Train/test splits saved locally.")

# ── Step 6: Upload splits back to Hugging Face ────────────────
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"   ✅ Uploaded {fname}")

print("\n🚀 All splits uploaded to Hugging Face Dataset Hub.")
