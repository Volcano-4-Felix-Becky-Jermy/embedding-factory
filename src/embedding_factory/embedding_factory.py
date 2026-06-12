import os


def get_embedding_model():
    """
    Factory function to get embedding model.
    embedding_type: "ollama" or "openai"
    model: model name (optional)
    """
    embedding_type = os.getenv("EMBEDDING_TYPE", "ollama")

    if embedding_type == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = os.getenv("OLLAMA_MODEL", "qwen3.5")
        return OllamaEmbeddings(model=model)
    elif embedding_type == "openrouter":
        from langchain_openai import OpenAIEmbeddings

        openai_api_key = os.getenv("OPENROUTER_API_KEY")
        openai_api_base = "https://openrouter.ai/api/v1"
        model = os.getenv("OPENROUTER_MODEL", "qwen3.5")
        return OpenAIEmbeddings(
            model=model,
            api_key=openai_api_key,
            base_url=openai_api_base,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
