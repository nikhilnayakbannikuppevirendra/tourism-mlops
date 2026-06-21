
# hosting.py
# Purpose: Push deployment files to Hugging Face Space


from huggingface_hub import HfApi
import os

HF_USERNAME = "nikhilnayakbv"   # ← Replace with your HF username
SPACE_REPO  = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",   # local folder with Dockerfile, app.py, requirements.txt
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f" Deployment files uploaded to: https://huggingface.co/spaces/{SPACE_REPO}")
