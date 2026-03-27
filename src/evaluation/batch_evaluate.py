import argparse
import csv
import json
import os
import statistics
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

SURVEY_ASPECTS = {
    "coverage": "How comprehensively the survey covers the key subtopics, methods, and perspectives under the topic.",
    "structure": "How logically organized, coherent, and easy to follow the survey is.",
    "relevance": "How tightly the survey stays aligned with the stated topic and avoids off-topic discussion.",
    "technical_depth": "How substantive, precise, and technically informative the survey is for a research audience.",
}

HYPOTHESIS_ASPECTS = {
    "novelty": "How non-obvious and research-worthy the hypothesis is relative to the topic context.",
    "clarity": "How clearly the hypothesis, mechanism, and claim are stated.",
    "testability": "How falsifiable the hypothesis is and how actionable the proposed validation plan appears.",
    "grounding": "How well the hypothesis is supported by, connected to, and motivated by the available survey context.",
    "potential_impact": "How meaningful the hypothesis could be if validated.",
}

REQUIRED_COLUMNS = (
    "topic",
    "survey_path",
    "conditioned_hypothesis_path",
    "unconditioned_hypothesis_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate survey and hypothesis outputs with an LLM judge."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to a JSON/JSONL/CSV manifest. "
            "Each row/item needs topic plus either *_path or *_text fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="src/evaluation/output",
        help="Directory to write detailed JSON and CSV summaries.",
    )
    parser.add_argument(
        "--eval-model",
        "--model",
        dest="eval_model",
        default=os.getenv("EVALUATE_MODEL") or os.getenv("MODEL"),
        help=(
            "Judge model name. Defaults to EVALUATE_MODEL from .env, "
            "then falls back to MODEL."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge model.",
    )
    parser.add_argument(
        "--max-survey-chars",
        type=int,
        default=24000,
        help="Maximum survey characters sent to the judge after truncation.",
    )
    parser.add_argument(
        "--max-hypothesis-chars",
        type=int,
        default=10000,
        help="Maximum hypothesis characters sent to the judge after truncation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for each topic when model output is malformed or transiently fails.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Seconds to sleep between topics.",
    )
    return parser.parse_args()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")
    if not api_base:
        raise ValueError("OPENAI_API_BASE is missing.")
    return OpenAI(api_key=api_key, base_url=api_base)


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=4000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict academic evaluator. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Empty model response.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    candidates: List[str] = []
    stack = 0
    start_idx: Optional[int] = None
    for index, char in enumerate(cleaned):
        if char == "{":
            if stack == 0:
                start_idx = index
            stack += 1
        elif char == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    candidates.append(cleaned[start_idx : index + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract a valid JSON object from model output.")


def truncate_middle(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False

    keep_head = int(max_chars * 0.7)
    keep_tail = max_chars - keep_head
    truncated = (
        text[:keep_head].rstrip()
        + "\n\n[... truncated for evaluation ...]\n\n"
        + text[-keep_tail:].lstrip()
    )
    return truncated, True


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def read_manifest(input_path: Path) -> List[Dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [dict(item) for item in payload["items"]]
        raise ValueError("JSON manifest must be a list or an object with an 'items' list.")

    if suffix == ".jsonl":
        items: List[Dict[str, Any]] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError("Unsupported manifest format. Use .json, .jsonl, or .csv.")


def load_json_file(file_path: Path) -> Any:
    return json.loads(file_path.read_text(encoding="utf-8"))


def render_hypothesis_card(payload: Dict[str, Any]) -> str:
    sections: List[str] = []
    field_order = [
        ("title", "Title"),
        ("research_gap", "Research Gap"),
        ("hypothesis_statement", "Hypothesis Statement"),
        ("mechanism", "Mechanism"),
        ("test_plan", "Test Plan"),
        ("expected_signal", "Expected Signal"),
        ("evidence_reasoning", "Evidence Reasoning"),
    ]
    for key, label in field_order:
        value = normalize_text(payload.get(key))
        if value:
            sections.append(f"{label}: {value}")

    cited_papers = payload.get("cited_papers")
    if isinstance(cited_papers, list) and cited_papers:
        sections.append("Cited Papers: " + "; ".join(normalize_text(item) for item in cited_papers if normalize_text(item)))

    if not sections:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return "\n".join(sections)


def choose_best_hypothesis(candidate: Any) -> Tuple[str, str]:
    if isinstance(candidate, str):
        text = candidate.strip()
        if not text:
            raise ValueError("Hypothesis text is empty.")
        return text, "raw_text"

    if isinstance(candidate, list):
        if not candidate:
            raise ValueError("Hypothesis list is empty.")
        return choose_best_hypothesis(candidate[0])

    if not isinstance(candidate, dict):
        raise ValueError("Unsupported hypothesis payload format.")

    if normalize_text(candidate.get("hypothesis_statement")):
        return render_hypothesis_card(candidate), "direct_hypothesis"

    selected_candidates = candidate.get("selected_candidates")
    if isinstance(selected_candidates, list) and selected_candidates:
        return choose_best_hypothesis(selected_candidates[0])

    hypotheses = candidate.get("hypotheses")
    if isinstance(hypotheses, list) and hypotheses:
        return choose_best_hypothesis(hypotheses[0])

    ranked_hypotheses = candidate.get("ranked_hypotheses")
    if isinstance(ranked_hypotheses, list) and ranked_hypotheses:
        ranked_sorted = sorted(
            ranked_hypotheses,
            key=lambda item: (
                _safe_float(item.get("rank"), default=9999),
                -_safe_float(item.get("total_score"), default=-1),
            ),
        )
        return render_hypothesis_card(ranked_sorted[0]), "ranked_hypothesis"

    for key in ("content", "text", "response", "output"):
        if normalize_text(candidate.get(key)):
            return normalize_text(candidate[key]), key

    return json.dumps(candidate, ensure_ascii=False, indent=2), "json_fallback"


def load_text_payload(
    entry: Dict[str, Any],
    key_prefix: str,
    base_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    inline_key = f"{key_prefix}_text"
    path_key = f"{key_prefix}_path"

    if normalize_text(entry.get(inline_key)):
        return normalize_text(entry[inline_key]), {"source": "inline_text"}

    raw_path = normalize_text(entry.get(path_key))
    if not raw_path:
        raise ValueError(f"Missing either '{inline_key}' or '{path_key}'.")

    resolved_path = Path(raw_path)
    if not resolved_path.is_absolute():
        resolved_path = (base_dir / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if key_prefix == "survey":
        if suffix in {".md", ".txt"}:
            return resolved_path.read_text(encoding="utf-8"), {
                "source": "file_text",
                "path": str(resolved_path),
            }
        if suffix == ".json":
            payload = load_json_file(resolved_path)
            if isinstance(payload, dict):
                for key in ("content", "survey", "survey_content", "text"):
                    if normalize_text(payload.get(key)):
                        return normalize_text(payload[key]), {
                            "source": f"json:{key}",
                            "path": str(resolved_path),
                        }
            raise ValueError(f"Could not extract survey text from: {resolved_path}")
        raise ValueError(f"Unsupported survey file type: {resolved_path}")

    if key_prefix in {"conditioned_hypothesis", "unconditioned_hypothesis"}:
        if suffix in {".md", ".txt"}:
            return resolved_path.read_text(encoding="utf-8"), {
                "source": "file_text",
                "path": str(resolved_path),
            }
        if suffix == ".json":
            payload = load_json_file(resolved_path)
            hypothesis_text, extraction = choose_best_hypothesis(payload)
            return hypothesis_text, {
                "source": extraction,
                "path": str(resolved_path),
            }
        raise ValueError(f"Unsupported hypothesis file type: {resolved_path}")

    raise ValueError(f"Unsupported key prefix: {key_prefix}")


def build_prompt(
    topic: str,
    survey_text: str,
    conditioned_hypothesis: str,
    unconditioned_hypothesis: str,
) -> str:
    survey_aspects = "\n".join(
        f"- {name}: {description}" for name, description in SURVEY_ASPECTS.items()
    )
    hypothesis_aspects = "\n".join(
        f"- {name}: {description}" for name, description in HYPOTHESIS_ASPECTS.items()
    )
    return textwrap.dedent(
        f"""
        Evaluate the following three research artifacts for the topic "{topic}".

        Artifacts:
        1. An OptiSurvey-generated review.
        2. A hypothesis generated with access to the review.
        3. A hypothesis generated without access to the review.

        Scoring instructions:
        - Use integer scores only: 1, 2, 3, 4, or 5.
        - Be strict and calibrated.
        - Judge each artifact on its own merits.
        - For the two hypotheses, use the review as context to judge how well grounded and research-relevant they are.
        - Keep rationales short and concrete.

        Survey aspects:
        {survey_aspects}

        Hypothesis aspects:
        {hypothesis_aspects}

        Return JSON only with this exact top-level structure:
        {{
          "topic": "{topic}",
          "survey": {{
            "scores": {{
              "coverage": 1,
              "structure": 1,
              "relevance": 1,
              "technical_depth": 1
            }},
            "overall_score": 1,
            "strengths": ["..."],
            "weaknesses": ["..."],
            "summary": "..."
          }},
          "conditioned_hypothesis": {{
            "scores": {{
              "novelty": 1,
              "clarity": 1,
              "testability": 1,
              "grounding": 1,
              "potential_impact": 1
            }},
            "overall_score": 1,
            "strengths": ["..."],
            "weaknesses": ["..."],
            "summary": "..."
          }},
          "unconditioned_hypothesis": {{
            "scores": {{
              "novelty": 1,
              "clarity": 1,
              "testability": 1,
              "grounding": 1,
              "potential_impact": 1
            }},
            "overall_score": 1,
            "strengths": ["..."],
            "weaknesses": ["..."],
            "summary": "..."
          }},
          "comparison": {{
            "better_hypothesis": "conditioned|unconditioned|tie",
            "reason": "..."
          }}
        }}

        Topic:
        {topic}

        OptiSurvey review:
        ---
        {survey_text}
        ---

        Hypothesis generated with the review:
        ---
        {conditioned_hypothesis}
        ---

        Hypothesis generated without the review:
        ---
        {unconditioned_hypothesis}
        ---
        """
    ).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_score(value: Any) -> int:
    return max(1, min(5, _safe_int(value, default=1)))


def normalize_eval_section(
    section: Dict[str, Any],
    expected_aspects: Dict[str, str],
) -> Dict[str, Any]:
    scores = section.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    normalized_scores = {
        aspect: clamp_score(scores.get(aspect, 1)) for aspect in expected_aspects
    }
    strengths = section.get("strengths", [])
    weaknesses = section.get("weaknesses", [])
    if not isinstance(strengths, list):
        strengths = [normalize_text(strengths)] if normalize_text(strengths) else []
    if not isinstance(weaknesses, list):
        weaknesses = [normalize_text(weaknesses)] if normalize_text(weaknesses) else []

    return {
        "scores": normalized_scores,
        "overall_score": clamp_score(
            section.get(
                "overall_score",
                round(statistics.mean(normalized_scores.values())),
            )
        ),
        "strengths": [normalize_text(item) for item in strengths if normalize_text(item)],
        "weaknesses": [normalize_text(item) for item in weaknesses if normalize_text(item)],
        "summary": normalize_text(section.get("summary")),
    }


def evaluate_topic(
    client: OpenAI,
    model: str,
    entry: Dict[str, Any],
    base_dir: Path,
    max_survey_chars: int,
    max_hypothesis_chars: int,
    temperature: float,
    max_retries: int,
) -> Dict[str, Any]:
    topic = normalize_text(entry.get("topic"))
    if not topic:
        raise ValueError("Each manifest item must include 'topic'.")

    survey_text, survey_meta = load_text_payload(entry, "survey", base_dir)
    conditioned_text, conditioned_meta = load_text_payload(
        entry, "conditioned_hypothesis", base_dir
    )
    unconditioned_text, unconditioned_meta = load_text_payload(
        entry, "unconditioned_hypothesis", base_dir
    )

    survey_text, survey_truncated = truncate_middle(survey_text, max_survey_chars)
    conditioned_text, conditioned_truncated = truncate_middle(
        conditioned_text, max_hypothesis_chars
    )
    unconditioned_text, unconditioned_truncated = truncate_middle(
        unconditioned_text, max_hypothesis_chars
    )

    prompt = build_prompt(
        topic=topic,
        survey_text=survey_text,
        conditioned_hypothesis=conditioned_text,
        unconditioned_hypothesis=unconditioned_text,
    )
    raw_response = call_llm(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_retries=max_retries,
    )
    parsed = extract_json_object(raw_response)

    survey_eval = normalize_eval_section(parsed.get("survey", {}), SURVEY_ASPECTS)
    conditioned_eval = normalize_eval_section(
        parsed.get("conditioned_hypothesis", {}), HYPOTHESIS_ASPECTS
    )
    unconditioned_eval = normalize_eval_section(
        parsed.get("unconditioned_hypothesis", {}), HYPOTHESIS_ASPECTS
    )
    comparison = parsed.get("comparison", {})
    better_hypothesis = normalize_text(comparison.get("better_hypothesis")).lower()
    if better_hypothesis not in {"conditioned", "unconditioned", "tie"}:
        cond_score = conditioned_eval["overall_score"]
        uncond_score = unconditioned_eval["overall_score"]
        if cond_score > uncond_score:
            better_hypothesis = "conditioned"
        elif uncond_score > cond_score:
            better_hypothesis = "unconditioned"
        else:
            better_hypothesis = "tie"

    return {
        "topic": topic,
        "input_sources": {
            "survey": {
                **survey_meta,
                "truncated": survey_truncated,
                "char_count_sent": len(survey_text),
            },
            "conditioned_hypothesis": {
                **conditioned_meta,
                "truncated": conditioned_truncated,
                "char_count_sent": len(conditioned_text),
            },
            "unconditioned_hypothesis": {
                **unconditioned_meta,
                "truncated": unconditioned_truncated,
                "char_count_sent": len(unconditioned_text),
            },
        },
        "survey": survey_eval,
        "conditioned_hypothesis": conditioned_eval,
        "unconditioned_hypothesis": unconditioned_eval,
        "comparison": {
            "better_hypothesis": better_hypothesis,
            "reason": normalize_text(comparison.get("reason")),
        },
        "raw_model_response": raw_response,
    }


def flatten_result_row(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "topic": result["topic"],
        "survey_overall_score": result["survey"]["overall_score"],
        "conditioned_overall_score": result["conditioned_hypothesis"]["overall_score"],
        "unconditioned_overall_score": result["unconditioned_hypothesis"]["overall_score"],
        "better_hypothesis": result["comparison"]["better_hypothesis"],
    }
    for aspect, score in result["survey"]["scores"].items():
        row[f"survey_{aspect}"] = score
    for aspect, score in result["conditioned_hypothesis"]["scores"].items():
        row[f"conditioned_{aspect}"] = score
    for aspect, score in result["unconditioned_hypothesis"]["scores"].items():
        row[f"unconditioned_{aspect}"] = score
    return row


def write_csv(file_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_summary(results: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    winner_counts = {"conditioned": 0, "unconditioned": 0, "tie": 0}
    for result in results:
        winner_counts[result["comparison"]["better_hypothesis"]] += 1

    def mean_for(section_name: str, aspect_name: Optional[str] = None) -> float:
        values: List[float] = []
        for result in results:
            section = result[section_name]
            if aspect_name is None:
                values.append(_safe_float(section["overall_score"]))
            else:
                values.append(_safe_float(section["scores"][aspect_name]))
        return round(statistics.mean(values), 4) if values else 0.0

    return {
        "num_topics_succeeded": len(results),
        "num_topics_failed": len(failures),
        "winner_counts": winner_counts,
        "survey_average_scores": {
            "overall_score": mean_for("survey"),
            **{
                aspect: mean_for("survey", aspect)
                for aspect in SURVEY_ASPECTS
            },
        },
        "conditioned_hypothesis_average_scores": {
            "overall_score": mean_for("conditioned_hypothesis"),
            **{
                aspect: mean_for("conditioned_hypothesis", aspect)
                for aspect in HYPOTHESIS_ASPECTS
            },
        },
        "unconditioned_hypothesis_average_scores": {
            "overall_score": mean_for("unconditioned_hypothesis"),
            **{
                aspect: mean_for("unconditioned_hypothesis", aspect)
                for aspect in HYPOTHESIS_ASPECTS
            },
        },
        "average_conditioned_minus_unconditioned_overall": round(
            statistics.mean(
                result["conditioned_hypothesis"]["overall_score"]
                - result["unconditioned_hypothesis"]["overall_score"]
                for result in results
            ),
            4,
        )
        if results
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not args.eval_model:
        raise ValueError(
            "No judge model provided. Set EVALUATE_MODEL in .env or pass --eval-model."
        )

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(input_path)
    base_dir = input_path.parent
    client = get_client()

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(f"Loaded {len(manifest)} topics from {input_path}")
    print(f"Using judge model: {args.eval_model}")

    for index, entry in enumerate(manifest, start=1):
        topic = normalize_text(entry.get("topic")) or f"item_{index}"
        print(f"[{index}/{len(manifest)}] Evaluating: {topic}")
        try:
            result = evaluate_topic(
                client=client,
                model=args.eval_model,
                entry=entry,
                base_dir=base_dir,
                max_survey_chars=args.max_survey_chars,
                max_hypothesis_chars=args.max_hypothesis_chars,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            results.append(result)
            print(
                "  Done | survey={survey} conditioned={conditioned} unconditioned={unconditioned} winner={winner}".format(
                    survey=result["survey"]["overall_score"],
                    conditioned=result["conditioned_hypothesis"]["overall_score"],
                    unconditioned=result["unconditioned_hypothesis"]["overall_score"],
                    winner=result["comparison"]["better_hypothesis"],
                )
            )
        except Exception as exc:
            failure = {
                "topic": topic,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"  Failed | {exc}")
        if args.sleep_between_requests > 0 and index < len(manifest):
            time.sleep(args.sleep_between_requests)

    aggregate = build_aggregate_summary(results, failures)
    payload = {
        "config": {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "model": args.eval_model,
            "temperature": args.temperature,
            "max_survey_chars": args.max_survey_chars,
            "max_hypothesis_chars": args.max_hypothesis_chars,
            "max_retries": args.max_retries,
        },
        "aggregate_summary": aggregate,
        "results": results,
        "failures": failures,
    }

    json_path = output_dir / "evaluation_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_rows = [flatten_result_row(result) for result in results]
    if detail_rows:
        write_csv(output_dir / "evaluation_summary.csv", detail_rows)

    print(f"Finished. Successful topics: {len(results)} | Failed topics: {len(failures)}")
    print(f"Detailed results written to: {json_path}")
    if detail_rows:
        print(f"Tabular summary written to: {output_dir / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
