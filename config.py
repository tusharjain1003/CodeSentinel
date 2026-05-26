from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    groq_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "codesentinel-codeqwen-7b"
    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    finetuned_model_name: str = "codesentinel-codeqwen-7b"
    gpt4o_model_name: str = "gpt-4o"
    groq_model_name: str = "llama3-70b-8192"
    github_token: str = ""
    github_webhook_secret: str = "change-me"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/codesentinel"
    hf_token: str = ""
    wandb_api_key: str = ""
    api_key: str = ""
    rate_limit_per_minute: int = 30
    allow_memory_db_fallback: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
