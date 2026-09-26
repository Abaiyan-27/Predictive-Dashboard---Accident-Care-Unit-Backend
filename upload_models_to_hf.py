import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

def main():
    load_dotenv()
    repo_id = os.getenv("HF_MODEL_REPO")
    token = os.getenv("HF_TOKEN")
    local_dir = "app/ml_models"

    if not repo_id or not token:
        print("Error: HF_MODEL_REPO or HF_TOKEN not set in .env")
        return

    print(f"Uploading {local_dir} to Hugging Face repository {repo_id}...")
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Update models from local deployment",
        )
        print("Successfully uploaded models to Hugging Face!")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
    main()
