import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Config:
    """Centralized configuration manager for environment variables and API keys."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY", "")

    @classmethod
    def validate(cls) -> None:
        """Validate required environment variables at application startup."""
        missing = []
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")

        if missing:
            raise ValueError(
                f"Missing critical environment variable(s): {', '.join(missing)}. "
                "Please check your .env file."
            )


config = Config()
def cofig():
    print(f"\n TAVILY_API_KEY :{config.TAVILY_API_KEY}")
    print(f"\nGROQ_API_KEY :{config.GROQ_API_KEY}")
    print(f"\n WEATHER_API_KEY :{config.WEATHER_API_KEY}")


if __name__== "__main__":
    cofig()

  