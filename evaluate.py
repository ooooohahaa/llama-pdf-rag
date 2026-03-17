import argparse
from pathlib import Path

from rag_app.config import AppConfig
from rag_app.evaluation import build_eval_dataset, load_eval_dataset, run_eval, save_eval_results
from rag_app.pipeline import configure_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--dataset-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env(Path(__file__).resolve().parent)
    if args.dataset_path:
        config = AppConfig(
            project_dir=config.project_dir,
            llm_api_key=config.llm_api_key,
            llm_api_base=config.llm_api_base,
            llm_model=config.llm_model,
            embed_api_key=config.embed_api_key,
            embed_api_base=config.embed_api_base,
            embed_model=config.embed_model,
            pgvector_url=config.pgvector_url,
            pgvector_table=config.pgvector_table,
            embed_dim=config.embed_dim,
            eval_dataset_path=Path(args.dataset_path),
            eval_questions_per_chunk=config.eval_questions_per_chunk,
            eval_chunk_size=config.eval_chunk_size,
            eval_max_chunks=config.eval_max_chunks,
            eval_question_language=config.eval_question_language,
        )
    configure_settings(config)
    if args.rebuild_dataset or not config.eval_dataset_path.exists():
        dataset = build_eval_dataset(config)
        print(f"dataset generated: {config.eval_dataset_path}")
        print(f"examples: {len(dataset.examples)}")
    dataset = load_eval_dataset(config.eval_dataset_path)
    result_df, summary = run_eval(config, dataset)

    # Save results to file and plot
    output_dir = Path("eval_outputs")
    save_eval_results(result_df, summary, output_dir)
    print(f"Results saved to {output_dir}")

    print(result_df.to_string(index=False))
    print(f"rows={summary.rows}")
    print(f"avg_faithfulness={summary.avg_faithfulness:.4f}")
    print(f"avg_relevancy={summary.avg_relevancy:.4f}")
    print(f"avg_correctness={summary.avg_correctness:.4f}")


if __name__ == "__main__":
    main()
