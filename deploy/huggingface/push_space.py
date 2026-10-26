"""Create or update the Hugging Face Space from deploy/huggingface/ (needs HF_TOKEN with write access).

    python deploy/huggingface/push_space.py <user-or-org>/amharic-tts-studio
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
here = Path(__file__).parent
api.upload_folder(folder_path=str(here), repo_id=repo, repo_type="space", allow_patterns=["Dockerfile", "README.md"])
print(f"https://huggingface.co/spaces/{repo}  (building; the server URL is https://{repo.replace('/', '-').lower()}.hf.space)")
