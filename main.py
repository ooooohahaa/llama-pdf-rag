from pathlib import Path
import os
import argparse
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


def build_llm() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("missing API key, set LLM_API_KEY or OPENAI_API_KEY")
    llm_kwargs = {"model": model, "api_key": api_key}
    if api_base:
        llm_kwargs["api_base"] = api_base
    return OpenAI(**llm_kwargs)


def build_embed_model() -> OpenAIEmbedding:
    embed_api_key = os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    embed_api_base = os.getenv("EMBED_API_BASE") or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    if not embed_api_key:
        raise RuntimeError("missing embedding key, set EMBED_API_KEY or LLM_API_KEY")
    embed_kwargs = {"model": embed_model, "api_key": embed_api_key}
    if embed_api_base:
        embed_kwargs["api_base"] = embed_api_base
    return OpenAIEmbedding(**embed_kwargs)


def load_pdf_documents() -> list:
    pdf_paths = sorted(PROJECT_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF found in project directory")
    target_pdf = pdf_paths[0]
    documents = SimpleDirectoryReader(input_files=[str(target_pdf)]).load_data()
    if not documents:
        raise RuntimeError("No pages loaded from PDF")
    print(f"PDF: {target_pdf.name}")
    print(f"Loaded pages: {len(documents)}")
    return documents


def build_pgvector_store() -> PGVectorStore:
    connection_uri = os.getenv("PGVECTOR_URL")
    if not connection_uri:
        raise RuntimeError("missing PGVECTOR_URL")
    table_name = os.getenv("PGVECTOR_TABLE", "linux_server_book")
    embed_dim = int(os.getenv("EMBED_DIM", "1536"))
    parsed = urlparse(connection_uri)
    if not parsed.hostname or not parsed.path:
        raise RuntimeError("invalid PGVECTOR_URL")
    return PGVectorStore.from_params(
        database=parsed.path.lstrip("/"),
        host=parsed.hostname,
        password=unquote(parsed.password or ""),
        port=int(parsed.port or 5432),
        user=unquote(parsed.username or ""),
        table_name=table_name,
        embed_dim=embed_dim,
    )


def ingest_pdf_to_pgvector(vector_store: PGVectorStore) -> None:
    documents = load_pdf_documents()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("PDF ingestion to PGVector completed.")


def get_query_index(vector_store: PGVectorStore) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def print_sources(response) -> None:
    source_nodes = getattr(response, "source_nodes", None) or []
    if not source_nodes:
        return
    print("\nTop retrieved chunks:")
    for i, node in enumerate(source_nodes[:3], start=1):
        text = node.get_text().replace("\n", " ").strip()
        preview = text[:160]
        print(f"{i}. score={node.score:.4f} | {preview}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Settings.llm = build_llm()
    Settings.embed_model = build_embed_model()
    vector_store = build_pgvector_store()
    if args.rebuild:
        ingest_pdf_to_pgvector(vector_store)
    index = get_query_index(vector_store)
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("RAG pipeline ready.")
    print("Using PGVector as retrieval backend.")
    print("Input question to query the PDF knowledge base. Type 'exit' to quit.")
    while True:
        user_input = input("\nQuery: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        response = query_engine.query(user_input)
        print(f"\nAnswer:\n{response}")
        print_sources(response)


if __name__ == "__main__":
    main()
