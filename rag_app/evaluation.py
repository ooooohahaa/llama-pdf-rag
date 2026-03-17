from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
from llama_index.core import Settings
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset.rag import LabelledRagDataset
from llama_index.core.node_parser import SentenceSplitter

from rag_app.config import AppConfig
from rag_app.pipeline import get_query_index, load_pdf_documents


@dataclass
class EvalSummary:
    rows: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_correctness: float


def _fallback_reference(example) -> str:
    contexts = getattr(example, "reference_contexts", None) or []
    if contexts:
        return contexts[0][:500]
    return ""


def _needs_fallback(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return True
    return "cannot access" in text or "i'm sorry" in text or "cannot retrieve" in text


def _safe_mean(values) -> float:
    valid = [x for x in values if x is not None]
    if not valid:
        return 0.0
    return mean(valid)


def _build_question_gen_query(language: str) -> str:
    if language.startswith("zh"):
        return "请仅基于给定内容生成问题，问题必须使用中文，避免超出原文信息。"
    return "Generate questions strictly based on the provided context."


def build_eval_dataset(config: AppConfig) -> LabelledRagDataset:
    documents = load_pdf_documents(config.project_dir)
    overlap = min(128, max(0, config.eval_chunk_size // 8))
    splitter = SentenceSplitter(chunk_size=config.eval_chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    selected_nodes = nodes[: config.eval_max_chunks]
    if not selected_nodes:
        raise RuntimeError("no nodes available for evaluation dataset generation")
    generator = RagDatasetGenerator(
        selected_nodes,
        llm=Settings.llm,
        num_questions_per_chunk=config.eval_questions_per_chunk,
        question_gen_query=_build_question_gen_query(config.eval_question_language),
        show_progress=True,
    )
    dataset = generator.generate_dataset_from_nodes()
    for example in dataset.examples:
        if _needs_fallback(example.reference_answer):
            example.reference_answer = _fallback_reference(example)
    config.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_json(str(config.eval_dataset_path))
    return dataset


def load_eval_dataset(path: Path) -> LabelledRagDataset:
    if not path.exists():
        raise FileNotFoundError(f"evaluation dataset not found: {path}")
    return LabelledRagDataset.from_json(str(path))


def run_eval(config: AppConfig, dataset: LabelledRagDataset) -> tuple[pd.DataFrame, EvalSummary]:
    index = get_query_index(config)
    retriever = index.as_retriever(similarity_top_k=5)
    query_engine = index.as_query_engine(similarity_top_k=5)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
    correctness_evaluator = CorrectnessEvaluator(llm=Settings.llm)
    records: list[dict] = []
    for example in dataset.examples:
        source_nodes = retriever.retrieve(example.query)
        if not source_nodes:
            records.append(
                {
                    "query": example.query,
                    "reference_answer": example.reference_answer,
                    "response": "RETRIEVAL_FAILED",
                    "faithfulness_score": None,
                    "faithfulness_passing": False,
                    "relevancy_score": None,
                    "relevancy_passing": False,
                    "correctness_score": None,
                    "correctness_passing": False,
                }
            )
            continue
        response = query_engine.query(example.query)
        faith = faithfulness_evaluator.evaluate_response(query=example.query, response=response)
        rel = relevancy_evaluator.evaluate_response(query=example.query, response=response)
        corr = correctness_evaluator.evaluate(
            query=example.query,
            response=str(response),
            reference=example.reference_answer,
        )
        records.append(
            {
                "query": example.query,
                "reference_answer": example.reference_answer,
                "response": str(response),
                "faithfulness_score": faith.score,
                "faithfulness_passing": faith.passing,
                "relevancy_score": rel.score,
                "relevancy_passing": rel.passing,
                "correctness_score": corr.score,
                "correctness_passing": corr.passing,
            }
        )
    if not records:
        raise RuntimeError("evaluation dataset is empty")
    df = pd.DataFrame(records)
    summary = EvalSummary(
        rows=len(df),
        avg_faithfulness=_safe_mean(df["faithfulness_score"]),
        avg_relevancy=_safe_mean(df["relevancy_score"]),
        avg_correctness=_safe_mean(df["correctness_score"]),
    )
    return df, summary


def save_eval_results(df: pd.DataFrame, summary: EvalSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_dir / "eval_results.csv", index=False)
    
    # Save Summary
    with open(output_dir / "eval_summary.txt", "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"Rows: {summary.rows}\n")
        f.write(f"Avg Faithfulness: {summary.avg_faithfulness:.4f}\n")
        f.write(f"Avg Relevancy: {summary.avg_relevancy:.4f}\n")
        f.write(f"Avg Correctness: {summary.avg_correctness:.4f}\n")
        
    # Plot
    plt.figure(figsize=(10, 6))
    metrics = ["faithfulness_score", "relevancy_score", "correctness_score"]
    # Drop NaNs for plotting
    plot_data = df[metrics].dropna()
    plot_data.boxplot()
    plt.title("RAG Evaluation Metrics Distribution")
    plt.ylabel("Score")
    plt.savefig(output_dir / "eval_metrics_distribution.png")
    plt.close()
