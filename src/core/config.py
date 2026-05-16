from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "model-example.h5"
    model_input_size: int = 150
    model_classes: list[str] = ["cloudy", "desert", "green_area", "water"]

    class Config:
        env_file = ".env"


settings = Settings()
