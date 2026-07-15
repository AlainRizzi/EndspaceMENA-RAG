from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    anthropic_api_key: str
    voyage_api_key: str
    llm_model: str = "..."
    embedding_model: str = "voyage-3"
    embedding_dimensions: int = 1024

    class Config:
        env_file = ".env"


settings = Settings()
