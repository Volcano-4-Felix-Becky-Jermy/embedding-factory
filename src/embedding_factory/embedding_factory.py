import os


def get_embedding_model():
    """
    Factory function to get embedding model.
    embedding_llm_provider: "ollama" or "openai"
    model: model name (optional)
    """
    embedding_llm_provider = os.getenv("EMBEDDING_LLM_PROVIDER", "ollama")

    if embedding_llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = os.getenv("OLLAMA_MODEL", "qwen3.5")
        return OllamaEmbeddings(model=model)
    elif embedding_llm_provider == "openrouter":
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

def serialize_retrieved_docs(docs):
    blocks = []
    for doc in docs:
        metadata = doc.metadata
        source = metadata.get("source", "unknown")
        platform = metadata.get("platform", source)
        author = metadata.get("author", "")
        url = metadata.get("url", "")
        created_at = metadata.get("created_at") or metadata.get("timestamp", "")
        title = metadata.get("title", "")

        source_line = f"Source: {platform} ({source})"
        if author:
            source_line += f" | Author: {author}"
        if created_at:
            source_line += f" | Created: {created_at}"
        if url:
            source_line += f" | URL: {url}"
        if title:
            source_line += f"\nTitle: {title}"

        image_summaries = metadata.get("image_summaries") or []
        image_summary_text = format_image_summaries(image_summaries)
        content = f"{source_line}\nContent: {doc.page_content}"
        if image_summary_text:
            content += f"\n{image_summary_text}"
        blocks.append(content)
    return "\n\n".join(blocks)

def format_image_summaries(image_summaries):
    if not image_summaries:
        return ""
    lines = ["Image Summaries:"]
    for index, item in enumerate(image_summaries, start=1):
        if isinstance(item, dict):
            summary = item.get("summary", "")
        else:
            summary = str(item)
        if summary:
            lines.append(f"- Image {index}: {summary}")
    return "\n".join(lines) if len(lines) > 1 else ""