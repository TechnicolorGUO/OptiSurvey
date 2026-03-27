import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DEMO_DIR = SRC_DIR / "demo"
SURVEY_PIPELINE_DIR = SRC_DIR / "demo" / "survey_generation_pipeline"

for path in (str(SRC_DIR), str(DEMO_DIR), str(SURVEY_PIPELINE_DIR), str(Path(__file__).resolve().parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "asg.settings")
from batch_evaluate import (  # noqa: E402
    build_aggregate_summary,
    evaluate_topic,
    flatten_result_row,
    normalize_text,
    write_csv,
)


_PIPELINE_IMPORTS: Optional[Tuple[Any, Any, Any]] = None
_DJANGO_IMPORTS: Optional[Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]] = None


def get_pipeline_imports() -> Tuple[Any, Any, Any]:
    global _PIPELINE_IMPORTS
    if _PIPELINE_IMPORTS is None:
        import importlib

        pipeline_main = importlib.import_module("survey_generation_pipeline.main")
        pipeline_retriever = importlib.import_module("survey_generation_pipeline.asg_retriever")
        pipeline_generator = importlib.import_module("survey_generation_pipeline.asg_generator")

        _PIPELINE_IMPORTS = (
            pipeline_main.ASG_system,
            pipeline_retriever.Retriever,
            pipeline_generator.getQwenClient,
        )
    return _PIPELINE_IMPORTS


def get_django_imports() -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    global _DJANGO_IMPORTS
    if _DJANGO_IMPORTS is None:
        import django

        django.setup()

        from django.test import RequestFactory
        from demo.views import (  # type: ignore
            _attach_hypothesis_evidence,
            _build_idea_evidence_packs,
            _extract_json_payload,
            _merge_autoresearch_rankings,
            download_pdfs_sync,
            generate_arxiv_query,
            run_autoresearch_sync,
        )
        from asg_generator import generateResponse  # type: ignore

        _DJANGO_IMPORTS = (
            RequestFactory,
            _attach_hypothesis_evidence,
            _build_idea_evidence_packs,
            _extract_json_payload,
            _merge_autoresearch_rankings,
            download_pdfs_sync,
            generate_arxiv_query,
            run_autoresearch_sync,
            generateResponse,
        )
    return _DJANGO_IMPORTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run survey generation, conditioned/unconditioned hypothesis generation, and evaluation."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON/JSONL/CSV topic manifest.")
    parser.add_argument(
        "--output-dir",
        default="src/evaluation/full_pipeline_output",
        help="Directory for copied artifacts and summaries.",
    )
    parser.add_argument(
        "--cluster-standard",
        default="research method",
        help="Cluster standard passed into OptiSurvey generation.",
    )
    parser.add_argument("--idea-count", type=int, default=5, help="Idea pool size for each hypothesis run.")
    parser.add_argument("--candidate-count", type=int, default=1, help="Final kept hypotheses per run.")
    parser.add_argument(
        "--skip-survey-generation",
        action="store_true",
        help="Reuse existing survey_json_path and citation_json_path from the manifest.",
    )
    parser.add_argument("--skip-evaluate", action="store_true", help="Generate artifacts only.")
    parser.add_argument("--sleep-between-topics", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def read_manifest(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [dict(item) for item in payload["items"]]
    elif suffix == ".jsonl":
        rows = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    elif suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("Unsupported manifest format. Use .json, .jsonl, or .csv.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify_topic(topic: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower()).strip("-")
    digest = hashlib.md5(topic.encode("utf-8")).hexdigest()[:10]
    return f"{base[:48]}-{digest}" if base else digest


def render_survey_markdown(survey_payload: Dict[str, Any]) -> str:
    title = normalize_text(survey_payload.get("survey_title") or survey_payload.get("topic"))
    content = normalize_text(survey_payload.get("content"))
    heading = f"# A Survey of {title}\n\n" if title else ""
    return heading + content + ("\n" if content else "")


def parse_json_response(response: Any) -> Dict[str, Any]:
    return json.loads(response.content.decode("utf-8"))


def copy_to_static_info(survey_id: str, citation_source: Path) -> Path:
    target_dir = ensure_dir(REPO_ROOT / "src" / "static" / "data" / "info" / survey_id)
    target_path = target_dir / "citation_data.json"
    shutil.copy2(citation_source, target_path)
    return target_path


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)


def normalize_reference_source(source: str) -> str:
    normalized = normalize_text(source).lower()
    if normalized in {"local", "json", "json-db", "json_vec", "vector_db", "vectordb"}:
        return "json_vec"
    if normalized == "hybrid":
        return "hybrid"
    return "arxiv"


def search_and_download_references(
    topic: str,
    source: str,
    pdf_root: Path,
    json_folder: Optional[str],
    local_db_path: Optional[str],
) -> Dict[str, Any]:
    RequestFactoryCls, _, _, _, _, download_pdfs_sync_fn, generate_arxiv_query_fn, _, _ = get_django_imports()
    factory = RequestFactoryCls()
    normalized_source = normalize_reference_source(source)

    query_payload: Dict[str, Any] = {
        "topic": topic,
        "source": normalized_source,
    }
    if json_folder:
        query_payload["json_folder"] = json_folder
    if local_db_path:
        query_payload["local_db_path"] = local_db_path

    query_request = factory.post(
        "/generate_arxiv_query/",
        data=json.dumps(query_payload),
        content_type="application/json",
    )
    query_response = generate_arxiv_query_fn(query_request)
    query_result = parse_json_response(query_response)
    if query_response.status_code >= 400:
        raise RuntimeError(query_result.get("error", f"Reference search failed for source={normalized_source}"))

    papers = query_result.get("papers", [])
    if not isinstance(papers, list) or not papers:
        raise ValueError(f"No references found for topic '{topic}' with source '{normalized_source}'.")

    pdf_links = [paper.get("pdf_link") for paper in papers if paper.get("pdf_link")]
    pdf_titles = [paper.get("title") or f"paper_{index}" for index, paper in enumerate(papers)]
    pdf_sources = [paper.get("source") or normalized_source for paper in papers]

    download_payload: Dict[str, Any] = {
        "pdf_links": pdf_links,
        "pdf_titles": pdf_titles,
        "sources": pdf_sources,
    }
    if json_folder:
        download_payload["json_folder"] = json_folder
    if local_db_path:
        download_payload["local_db_path"] = local_db_path

    download_request = factory.post(
        "/download_pdfs/",
        data=json.dumps(download_payload),
        content_type="application/json",
    )
    download_response = download_pdfs_sync_fn(download_request, operation_id=f"offline_download_{int(time.time())}")
    download_result = parse_json_response(download_response)
    if download_response.status_code >= 400:
        raise RuntimeError(download_result.get("message") or download_result.get("error") or "PDF download failed")

    copied_files = []
    ensure_dir(pdf_root)
    for file_path in download_result.get("files", []):
        source_path = Path(file_path).resolve()
        if not source_path.exists():
            continue
        destination = pdf_root / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(str(destination))

    if not copied_files:
        raise RuntimeError("Reference search succeeded but no PDFs were copied into the working directory.")

    return {
        "source": normalized_source,
        "papers_found": len(papers),
        "downloaded_files": copied_files,
        "failed_downloads": download_result.get("failed", []),
        "search_response": query_result,
    }


def generate_survey(
    topic: str,
    survey_id: str,
    topic_dir: Path,
    cluster_standard: str,
    pdf_dir: str | None,
    auto_download: bool,
    source: str,
    json_folder: Optional[str],
    local_db_path: Optional[str],
) -> Dict[str, Any]:
    ASGSystem, RetrieverCls, _ = get_pipeline_imports()
    pdf_root = Path(pdf_dir).resolve() if pdf_dir else ensure_dir(topic_dir / "pdfs")
    ensure_dir(REPO_ROOT / "src" / "static" / "data" / "txt" / survey_id)
    ensure_dir(REPO_ROOT / "info" / survey_id)

    runtime: Dict[str, Any] = {}
    retriever = RetrieverCls()
    asg_system = ASGSystem(".", survey_id, str(pdf_root), topic, cluster_standard)

    if auto_download:
        start = time.time()
        reference_download_result = search_and_download_references(
            topic=topic,
            source=source,
            pdf_root=pdf_root,
            json_folder=json_folder,
            local_db_path=local_db_path,
        )
        runtime["download_pdf_seconds"] = round(time.time() - start, 2)
        runtime["reference_source"] = reference_download_result["source"]
        runtime["papers_found"] = reference_download_result["papers_found"]
        runtime["downloaded_pdf_count"] = len(reference_download_result["downloaded_files"])
        runtime["failed_download_count"] = len(reference_download_result.get("failed_downloads", []))

    start = time.time()
    asg_system.parsing_pdfs()
    runtime["parsing_pdfs_seconds"] = round(time.time() - start, 2)

    start = time.time()
    asg_system.description_generation(retriever)
    runtime["description_generation_seconds"] = round(time.time() - start, 2)

    start = time.time()
    asg_system.agglomerative_clustering()
    runtime["agglomerative_clustering_seconds"] = round(time.time() - start, 2)

    start = time.time()
    asg_system.outline_generation()
    runtime["outline_generation_seconds"] = round(time.time() - start, 2)

    start = time.time()
    asg_system.section_generation()
    runtime["section_generation_seconds"] = round(time.time() - start, 2)

    survey_json_source = REPO_ROOT / "src" / "static" / "data" / "txt" / survey_id / "generated_result.json"
    citation_json_source = REPO_ROOT / "info" / survey_id / "citation_data.json"
    if not survey_json_source.exists():
        raise FileNotFoundError(f"Survey output not found: {survey_json_source}")
    if not citation_json_source.exists():
        raise FileNotFoundError(f"Citation data not found: {citation_json_source}")

    static_citation_path = copy_to_static_info(survey_id, citation_json_source)
    survey_dir = ensure_dir(topic_dir / "survey")
    survey_json_target = survey_dir / "generated_result.json"
    citation_json_target = survey_dir / "citation_data.json"
    survey_md_target = survey_dir / "survey.md"
    runtime_target = survey_dir / "runtime.json"

    shutil.copy2(survey_json_source, survey_json_target)
    shutil.copy2(static_citation_path, citation_json_target)
    survey_payload = read_json(survey_json_target)
    survey_md_target.write_text(render_survey_markdown(survey_payload), encoding="utf-8")
    write_json(runtime_target, runtime)
    copy_if_exists(REPO_ROOT / "info" / survey_id / "outline.json", survey_dir / "outline.json")
    copy_if_exists(REPO_ROOT / "info" / survey_id / "cluster_info.json", survey_dir / "cluster_info.json")

    return {
        "survey_json_path": str(survey_json_target),
        "survey_markdown_path": str(survey_md_target),
        "citation_json_path": str(citation_json_target),
        "runtime_json_path": str(runtime_target),
    }


def run_conditioned_hypothesis(survey_id: str, idea_count: int, candidate_count: int) -> Dict[str, Any]:
    RequestFactoryCls, _, _, _, _, _, _, run_autoresearch_sync_fn, _ = get_django_imports()
    factory = RequestFactoryCls()
    request = factory.post(
        "/run_autoresearch/",
        data=json.dumps(
            {
                "survey_id": survey_id,
                "cycle_index": 1,
                "max_iterations": 2,
                "idea_count": idea_count,
                "candidate_count": candidate_count,
                "history": [],
            }
        ),
        content_type="application/json",
    )
    response = run_autoresearch_sync_fn(request, operation_id=f"offline_autoresearch_{survey_id}")
    payload = parse_json_response(response)
    if response.status_code >= 400:
        raise RuntimeError(payload.get("error", "Conditioned hypothesis generation failed"))
    return payload


def build_unconditioned_brainstorm_prompt(topic: str, citation_data: List[Dict[str, Any]], idea_count: int) -> str:
    evidence_context = json.dumps(citation_data[:8], ensure_ascii=False, indent=2)
    return f"""
You are the Brainstorming Agent in AutoResearch.
Generate exactly {idea_count} research ideas for the topic below without using any generated survey review.

Rules:
- Work only from the topic and the uploaded-paper evidence.
- Keep each idea concrete and distinct.
- Return JSON only in this exact shape:
{{
  "ideas": [
    {{
      "id": "I1",
      "title": "Short title",
      "core_insight": "One concise paragraph",
      "novelty_basis": "Why this seems new",
      "why_now": "Why this matters now"
    }}
  ]
}}

Topic: {topic}

Uploaded-paper evidence:
{evidence_context}
""".strip()


def build_unconditioned_hypothesis_prompt(topic: str, evidence_packs: List[Dict[str, Any]]) -> str:
    return f"""
You are the Hypothesis Agent in AutoResearch.
Transform each brainstormed idea into one structured, falsifiable hypothesis.
Use the uploaded-paper evidence packs for grounding when evidence exists.

Rules:
- Produce exactly one hypothesis per idea.
- Use only citation source names that appear inside each idea's evidence pack.
- Keep the fields concise and specific.
- Return JSON only in this exact shape:
{{
  "hypotheses": [
    {{
      "idea_id": "I1",
      "hypothesis_id": "H1",
      "title": "Card title",
      "research_gap": "Gap statement",
      "hypothesis_statement": "If ..., then ... because ...",
      "mechanism": "Underlying logic",
      "test_plan": "How to test it",
      "expected_signal": "What result would support it",
      "evidence_reasoning": "How the uploaded papers motivate it",
      "cited_papers": ["Paper A", "Paper B"]
    }}
  ]
}}

Topic: {topic}

Idea and evidence packs:
{json.dumps(evidence_packs, ensure_ascii=False, indent=2)}
""".strip()


def build_validation_prompt(hypotheses: List[Dict[str, Any]], candidate_count: int, topic: str) -> str:
    return f"""
You are the Validation Agent in AutoResearch, acting as a rigorous reviewer.
Review the hypotheses and rank them for a single research cycle.

Scoring dimensions:
- novelty: 1-10
- literature_grounding: 1-10
- clarity: 1-10
- potential_impact: 1-10
- total_score: 0-100

Rules:
- Rank all hypotheses from strongest to weakest.
- Select the best {candidate_count} candidates.
- Keep reviewer summaries short and specific.
- Return JSON only in this exact shape:
{{
  "review_summary": "Overall summary",
  "continue_recommendation": "stop",
  "continue_reason": "One short reason",
  "ranked_hypotheses": [
    {{
      "hypothesis_id": "H1",
      "rank": 1,
      "novelty": 8,
      "literature_grounding": 7,
      "clarity": 9,
      "potential_impact": 8,
      "total_score": 82,
      "reviewer_summary": "Short reviewer comment"
    }}
  ],
  "selected_candidate_ids": ["H1"]
}}

Topic: {topic}

Structured hypotheses:
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}
""".strip()


def run_unconditioned_hypothesis(topic: str, citation_data: List[Dict[str, Any]], idea_count: int, candidate_count: int, max_retries: int) -> Dict[str, Any]:
    _, _, get_qwen_client = get_pipeline_imports()
    _, attach_hypothesis_evidence, build_idea_evidence_packs, extract_json_payload, merge_autoresearch_rankings, _, _, _, generate_response = get_django_imports()
    client = get_qwen_client()
    brainstorming_raw = generate_response(client, build_unconditioned_brainstorm_prompt(topic, citation_data, idea_count), max_retries=max_retries)
    ideas = extract_json_payload(brainstorming_raw, root_key="ideas").get("ideas", [])
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("Unconditioned brainstorming returned no ideas")

    evidence_packs = build_idea_evidence_packs(ideas[:idea_count], citation_data)
    hypothesis_raw = generate_response(client, build_unconditioned_hypothesis_prompt(topic, evidence_packs), max_retries=max_retries)
    hypotheses = extract_json_payload(hypothesis_raw, root_key="hypotheses").get("hypotheses", [])
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("Unconditioned hypothesis generation returned no hypotheses")
    hypotheses = attach_hypothesis_evidence(hypotheses[: len(evidence_packs)], evidence_packs)

    validation_raw = generate_response(client, build_validation_prompt(hypotheses, candidate_count, topic), max_retries=max_retries)
    validation_payload = extract_json_payload(validation_raw)
    ranked_hypotheses = validation_payload.get("ranked_hypotheses", [])
    selected_candidate_ids = validation_payload.get("selected_candidate_ids", [])
    if isinstance(selected_candidate_ids, str):
        selected_candidate_ids = [selected_candidate_ids]
    if not isinstance(selected_candidate_ids, list):
        selected_candidate_ids = []

    merged_rankings, selected_candidates = merge_autoresearch_rankings(
        hypotheses, ranked_hypotheses, selected_candidate_ids[:candidate_count]
    )
    if not selected_candidates:
        selected_candidates = merged_rankings[:candidate_count]

    return {
        "success": True,
        "survey_title": topic,
        "cycle": {
            "iteration": 1,
            "execution_mode": "manual",
            "idea_count_requested": idea_count,
            "candidate_count_requested": candidate_count,
            "ideas": ideas[:idea_count],
            "hypotheses": hypotheses,
            "ranked_hypotheses": merged_rankings,
            "selected_candidates": selected_candidates,
            "review_summary": str(validation_payload.get("review_summary") or "").strip(),
            "continue_recommendation": str(validation_payload.get("continue_recommendation") or "").strip().lower(),
            "continue_reason": str(validation_payload.get("continue_reason") or "").strip(),
        },
        "raw_model_responses": {
            "brainstorming": brainstorming_raw,
            "hypothesis": hypothesis_raw,
            "validation": validation_raw,
        },
    }


def process_topic(entry: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    topic = normalize_text(entry.get("topic"))
    if not topic:
        raise ValueError("Each manifest item must contain 'topic'.")

    survey_id = normalize_text(entry.get("survey_id")) or slugify_topic(topic)
    topic_dir = ensure_dir(output_dir / survey_id)

    if args.skip_survey_generation:
        survey_json_path = Path(normalize_text(entry.get("survey_json_path"))).resolve()
        citation_json_path = Path(normalize_text(entry.get("citation_json_path"))).resolve()
        if not survey_json_path.exists():
            raise FileNotFoundError(f"survey_json_path not found: {survey_json_path}")
        if not citation_json_path.exists():
            raise FileNotFoundError(f"citation_json_path not found: {citation_json_path}")
        survey_dir = ensure_dir(topic_dir / "survey")
        shutil.copy2(survey_json_path, survey_dir / "generated_result.json")
        shutil.copy2(citation_json_path, survey_dir / "citation_data.json")
        copy_to_static_info(survey_id, citation_json_path)
        survey_payload = read_json(survey_dir / "generated_result.json")
        (survey_dir / "survey.md").write_text(render_survey_markdown(survey_payload), encoding="utf-8")
    else:
        generate_survey(
            topic=topic,
            survey_id=survey_id,
            topic_dir=topic_dir,
            cluster_standard=args.cluster_standard,
            pdf_dir=normalize_text(entry.get("pdf_dir")) or None,
            auto_download=str(entry.get("auto_download", "true")).strip().lower() not in {"false", "0", "no"},
            source=normalize_text(entry.get("source")) or "arxiv",
            json_folder=normalize_text(entry.get("json_folder")) or None,
            local_db_path=normalize_text(entry.get("local_db_path")) or None,
        )

    survey_json = read_json(topic_dir / "survey" / "generated_result.json")
    citation_json = read_json(topic_dir / "survey" / "citation_data.json")

    conditioned_payload = run_conditioned_hypothesis(
        survey_id=survey_id,
        idea_count=args.idea_count,
        candidate_count=args.candidate_count,
    )
    unconditioned_payload = run_unconditioned_hypothesis(
        topic=topic,
        citation_data=citation_json if isinstance(citation_json, list) else [],
        idea_count=args.idea_count,
        candidate_count=args.candidate_count,
        max_retries=args.max_retries,
    )

    hypotheses_dir = ensure_dir(topic_dir / "hypotheses")
    conditioned_path = hypotheses_dir / "conditioned_hypothesis.json"
    unconditioned_path = hypotheses_dir / "unconditioned_hypothesis.json"
    write_json(conditioned_path, conditioned_payload)
    write_json(unconditioned_path, unconditioned_payload)

    result: Dict[str, Any] = {
        "topic": topic,
        "survey_id": survey_id,
        "artifact_paths": {
            "survey_json_path": str(topic_dir / "survey" / "generated_result.json"),
            "survey_markdown_path": str(topic_dir / "survey" / "survey.md"),
            "citation_json_path": str(topic_dir / "survey" / "citation_data.json"),
            "conditioned_hypothesis_path": str(conditioned_path),
            "unconditioned_hypothesis_path": str(unconditioned_path),
        },
    }

    if not args.skip_evaluate:
        _, _, get_qwen_client = get_pipeline_imports()
        result["evaluation"] = evaluate_topic(
            client=get_qwen_client(),
            model=os.getenv("EVALUATE_MODEL") or os.getenv("MODEL") or "",
            entry={
                "topic": topic,
                "survey_path": result["artifact_paths"]["survey_json_path"],
                "conditioned_hypothesis_path": result["artifact_paths"]["conditioned_hypothesis_path"],
                "unconditioned_hypothesis_path": result["artifact_paths"]["unconditioned_hypothesis_path"],
            },
            base_dir=REPO_ROOT,
            max_survey_chars=24000,
            max_hypothesis_chars=10000,
            temperature=0.0,
            max_retries=3,
        )

    write_json(topic_dir / "topic_result.json", result)
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    manifest = read_manifest(input_path)

    if not manifest:
        raise ValueError("No topics found in the manifest.")
    if not os.getenv("MODEL"):
        raise ValueError("MODEL is required for generation.")
    if not args.skip_evaluate and not (os.getenv("EVALUATE_MODEL") or os.getenv("MODEL")):
        raise ValueError("EVALUATE_MODEL or MODEL is required for evaluation.")

    results = []
    failures = []

    print(f"Loaded {len(manifest)} topics from {input_path}")
    print(f"Generation model: {os.getenv('MODEL')}")
    if not args.skip_evaluate:
        print(f"Evaluation model: {os.getenv('EVALUATE_MODEL') or os.getenv('MODEL')}")

    for index, entry in enumerate(manifest, start=1):
        topic = normalize_text(entry.get("topic")) or f"item_{index}"
        print(f"[{index}/{len(manifest)}] Processing: {topic}")
        try:
            result = process_topic(entry, args, output_dir)
            results.append(result)
            if "evaluation" in result:
                evaluation = result["evaluation"]
                print(
                    "  Done | survey={survey} conditioned={conditioned} unconditioned={unconditioned} winner={winner}".format(
                        survey=evaluation["survey"]["overall_score"],
                        conditioned=evaluation["conditioned_hypothesis"]["overall_score"],
                        unconditioned=evaluation["unconditioned_hypothesis"]["overall_score"],
                        winner=evaluation["comparison"]["better_hypothesis"],
                    )
                )
            else:
                print("  Done | generation only")
        except Exception as exc:
            failures.append({"topic": topic, "error": str(exc)})
            print(f"  Failed | {exc}")

        if args.sleep_between_topics > 0 and index < len(manifest):
            time.sleep(args.sleep_between_topics)

    payload: Dict[str, Any] = {
        "config": {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "cluster_standard": args.cluster_standard,
            "idea_count": args.idea_count,
            "candidate_count": args.candidate_count,
            "skip_survey_generation": args.skip_survey_generation,
            "skip_evaluate": args.skip_evaluate,
            "generation_model": os.getenv("MODEL"),
            "evaluation_model": os.getenv("EVALUATE_MODEL") or os.getenv("MODEL"),
        },
        "results": results,
        "failures": failures,
    }

    if not args.skip_evaluate:
        evaluation_results = [result["evaluation"] for result in results if "evaluation" in result]
        payload["aggregate_summary"] = build_aggregate_summary(evaluation_results, failures)
        detail_rows = [flatten_result_row(result["evaluation"]) for result in results if "evaluation" in result]
        if detail_rows:
            write_csv(output_dir / "evaluation_summary.csv", detail_rows)

    write_json(output_dir / "full_pipeline_results.json", payload)
    print(f"Finished. Success={len(results)} Failed={len(failures)}")
    print(f"Summary written to: {output_dir / 'full_pipeline_results.json'}")


if __name__ == "__main__":
    main()
