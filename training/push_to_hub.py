from huggingface_hub import HfApi


def push(local_path: str, repo_id: str) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")
