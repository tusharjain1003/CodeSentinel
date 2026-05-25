from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "codesentinel-codeqwen-7b"
    github_token: str = ""
    github_webhook_secret: str = "change-me"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/codesentinel"
    hf_token: str = ""
    wandb_api_key: str = ""
    api_key: str = ""
    rate_limit_per_minute: int = 30

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
