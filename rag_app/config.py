from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore


@dataclass(frozen=True)
class AppConfig:
    project_dir: Path
    llm_api_key: str
    llm_api_base: str | None
    llm_model: str
    embed_api_key: str
    embed_api_base: str | None
    embed_model: str
    pgvector_url: str
    pgvector_table: str
    embed_dim: int
    eval_dataset_path: Path
    eval_questions_per_chunk: int
    eval_chunk_size: int
    eval_max_chunks: int
    eval_question_language: str

    @staticmethod
    def from_env(project_dir: Path) -> "AppConfig":
        load_dotenv(project_dir / ".env")
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        llm_api_base = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        embed_api_key = os.getenv("EMBED_API_KEY") or llm_api_key
        embed_api_base = os.getenv("EMBED_API_BASE") or llm_api_base
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        pgvector_url = os.getenv("PGVECTOR_URL", "")
        pgvector_table = os.getenv("PGVECTOR_TABLE", "linux_server_book")
        embed_dim = int(os.getenv("EMBED_DIM", "1536"))
        eval_dataset_path = Path(os.getenv("EVAL_DATASET_PATH", str(project_dir / "eval_dataset.json")))
        eval_questions_per_chunk = int(os.getenv("EVAL_QUESTIONS_PER_CHUNK", "2"))
        eval_chunk_size = int(os.getenv("EVAL_CHUNK_SIZE", "512"))
        eval_max_chunks = int(os.getenv("EVAL_MAX_CHUNKS", "8"))
        eval_question_language = os.getenv("EVAL_QUESTION_LANGUAGE", "zh").strip().lower()
        if not llm_api_key:
            raise RuntimeError("missing API key, set LLM_API_KEY or OPENAI_API_KEY")
        if not embed_api_key:
            raise RuntimeError("missing embedding key, set EMBED_API_KEY or LLM_API_KEY")
        if not pgvector_url:
            raise RuntimeError("missing PGVECTOR_URL")
        return AppConfig(
            project_dir=project_dir,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            embed_api_key=embed_api_key,
            embed_api_base=embed_api_base,
            embed_model=embed_model,
            pgvector_url=pgvector_url,
            pgvector_table=pgvector_table,
            embed_dim=embed_dim,
            eval_dataset_path=eval_dataset_path,
            eval_questions_per_chunk=eval_questions_per_chunk,
            eval_chunk_size=eval_chunk_size,
            eval_max_chunks=eval_max_chunks,
            eval_question_language=eval_question_language,
        )


def build_llm(config: AppConfig) -> OpenAI:
    llm_kwargs = {"model": config.llm_model, "api_key": config.llm_api_key}
    if config.llm_api_base:
        llm_kwargs["api_base"] = config.llm_api_base
    return OpenAI(**llm_kwargs)


def build_embed_model(config: AppConfig) -> OpenAIEmbedding:
    embed_kwargs = {"model": config.embed_model, "api_key": config.embed_api_key}
    if config.embed_api_base:
        embed_kwargs["api_base"] = config.embed_api_base
    return OpenAIEmbedding(**embed_kwargs)


def build_pgvector_store(config: AppConfig) -> PGVectorStore:
    parsed = urlparse(config.pgvector_url)
    if not parsed.hostname or not parsed.path:
        raise RuntimeError("invalid PGVECTOR_URL")
    return PGVectorStore.from_params(
        database=parsed.path.lstrip("/"),
        host=parsed.hostname,
        password=unquote(parsed.password or ""),
        port=int(parsed.port or 5432),
        user=unquote(parsed.username or ""),
        table_name=config.pgvector_table,
        embed_dim=config.embed_dim,
    )
