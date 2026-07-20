from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://localhost:6379"
    gemini_api_key: str
    llm_model: str = "gemini-3.1-flash-lite"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    rerank_model: str = "cohere.rerank-v3-5:0"

    aws_access_key_id: str
    aws_access_key_secret: str
    aws_region: str = "ap-southeast-2"
    s3_bucket_name: str

    class Config:
        env_file = ".env"


settings = Settings()
