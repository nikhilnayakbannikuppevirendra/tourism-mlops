# ─────────────────────────────────────────────────────────────
# data_register.py
# Purpose: Register the raw dataset on Hugging Face Dataset Hub
# ─────────────────────────────────────────────────────────────

from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# ── Configuration ────────────────────────────────────────────
HF_USERNAME = "nikhilnayakbv"           # ← Replace with your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness"
REPO_TYPE = "dataset"

# ── Initialize HF API with token from environment ────────────
api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Step 1: Ensure the dataset repository exists ─────────────
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE)
    print(f"✅ Dataset repo '{DATASET_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"⚠️  Dataset repo '{DATASET_REPO_ID}' not found. Creating ...")
    create_repo(repo_id=DATASET_REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"✅ Dataset repo '{DATASET_REPO_ID}' created.")

# ── Step 2: Upload the data folder ───────────────────────────
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=DATASET_REPO_ID,
    repo_type=REPO_TYPE,
)
print("✅ Data folder uploaded to Hugging Face Dataset Hub successfully.")
