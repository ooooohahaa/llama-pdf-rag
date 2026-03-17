from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex

from rag_app.config import AppConfig, build_embed_model, build_llm, build_pgvector_store


def configure_settings(config: AppConfig) -> None:
    Settings.llm = build_llm(config)
    Settings.embed_model = build_embed_model(config)


def load_pdf_documents(project_dir: Path) -> list:
    pdf_paths = sorted(project_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF found in project directory")
    target_pdf = pdf_paths[0]
    documents = SimpleDirectoryReader(input_files=[str(target_pdf)]).load_data()
    if not documents:
        raise RuntimeError("No pages loaded from PDF")
    print(f"PDF: {target_pdf.name}")
    print(f"Loaded pages: {len(documents)}")
    return documents


def ingest_pdf_to_pgvector(config: AppConfig) -> None:
    vector_store = build_pgvector_store(config)
    documents = load_pdf_documents(config.project_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("PDF ingestion to PGVector completed.")


def get_query_index(config: AppConfig) -> VectorStoreIndex:
    vector_store = build_pgvector_store(config)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
