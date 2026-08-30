from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()


def pick_llm(level: str):

    models = {
        "low": "google/gemma-4-26b-a4b-it:free",
        "medium": "nvidia/nemotron-3-super-120b-a12b:free",
        "high": "nvidia/nemotron-3-ultra-550b-a55b:free",
    }

    level = level.lower()

    if level not in models:
        raise ValueError(
            f"Invalid level: {level}. "
            f"Choose from: {list(models.keys())}"
        )

    return ChatOpenRouter(
        model=models[level],
        temperature=0,
    )
