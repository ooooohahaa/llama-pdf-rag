import argparse
from pathlib import Path

from rag_app.config import AppConfig
from rag_app.pipeline import configure_settings, get_query_index, ingest_pdf_to_pgvector


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
    config = AppConfig.from_env(Path(__file__).resolve().parent)
    configure_settings(config)
    if args.rebuild:
        ingest_pdf_to_pgvector(config)
    index = get_query_index(config)
    retriever = index.as_retriever(similarity_top_k=5)
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
        source_nodes = retriever.retrieve(user_input)
        if not source_nodes:
            print("\n检索失败：知识库未命中相关内容。")
            continue
        response = query_engine.query(user_input)
        print(f"\nAnswer:\n{response}")
        print_sources(response)


if __name__ == "__main__":
    main()
