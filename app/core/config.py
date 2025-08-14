from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Speaker Detection Service"
    debug: bool = False

    class Config:
        env_file = ".env"


settings = Settings()

SERVICE_NAME = "AI Speaker Detection Service"
SERVICE_VERSION = "1.0.0"
