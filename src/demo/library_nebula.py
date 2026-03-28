from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from umap import UMAP
except ImportError:
    UMAP = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_ROOT = PROJECT_ROOT / "src" / "static" / "local_pdfs"
CACHE_PATH = PROJECT_ROOT / "src" / "static" / "data" / "cache" / "library_nebula.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_VERSION = "v2-clustered"
PALETTE = [
    "#7dd3fc",
    "#38bdf8",
    "#22d3ee",
    "#2dd4bf",
    "#a3e635",
    "#fbbf24",
    "#fb7185",
    "#f472b6",
    "#c084fc",
    "#818cf8",
    "#60a5fa",
    "#34d399",
]


def _resolve_json_root(root: str | Path | None = None) -> Path:
    if root is None:
        return DEFAULT_JSON_ROOT
    candidate = Path(root)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _compute_signature(json_files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    for path in json_files:
        stat = path.stat()
        try:
            path_key = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            path_key = str(path.resolve())
        digest.update(path_key.encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def _to_static_url(pdf_path: str) -> str:
    normalized = pdf_path.replace("\\", "/")
    if normalized.startswith("src/static/"):
        return "/static/" + normalized[len("src/static/") :]

    absolute = Path(pdf_path)
    if absolute.is_absolute():
        try:
            relative = absolute.resolve().relative_to((PROJECT_ROOT / "src" / "static").resolve())
            return "/static/" + relative.as_posix()
        except ValueError:
            return ""

    return ""


def _build_unique_title(seen: set[str], base_title: str, source_name: str) -> str:
    title = base_title.strip()
    if title not in seen:
        seen.add(title)
        return title

    suffix = 2
    candidate = f"{title} ({source_name})"
    while candidate in seen:
        suffix += 1
        candidate = f"{title} ({source_name} {suffix})"
    seen.add(candidate)
    return candidate


def _load_library_entries(json_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not json_root.exists():
        return [], []

    json_files = sorted(path for path in json_root.glob("*.json") if path.is_file())
    entries: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(json_file.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        source_name = json_file.stem
        for raw_title, raw_pdf_path in data.items():
            if not isinstance(raw_title, str) or not isinstance(raw_pdf_path, str):
                continue

            title = raw_title.strip()
            pdf_path = raw_pdf_path.strip()
            if not title or not pdf_path:
                continue

            unique_title = _build_unique_title(seen_titles, title, source_name)
            paper_id = hashlib.md5(f"{unique_title}|{pdf_path}|{source_name}".encode("utf-8")).hexdigest()[:16]
            entries.append(
                {
                    "id": f"nebula_{paper_id}",
                    "title": unique_title,
                    "display_title": title,
                    "pdf_path": pdf_path.replace("\\", "/"),
                    "pdf_url": _to_static_url(pdf_path),
                    "source": source_name,
                    "source_file": json_file.name,
                }
            )

    return entries, json_files


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        embeddings = np.atleast_2d(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _embed_titles(titles: list[str]) -> tuple[np.ndarray, str]:
    if not titles:
        return np.empty((0, 2), dtype=np.float32), "empty"

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)
        return _normalize_embeddings(np.asarray(embeddings, dtype=np.float32)), EMBEDDING_MODEL_NAME
    except Exception:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
        sparse_matrix = vectorizer.fit_transform(titles)
        if sparse_matrix.shape[0] <= 2 or sparse_matrix.shape[1] <= 2:
            embeddings = sparse_matrix.toarray().astype(np.float32)
        else:
            components = min(64, sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1)
            if components < 2:
                embeddings = sparse_matrix.toarray().astype(np.float32)
            else:
                embeddings = TruncatedSVD(n_components=components, random_state=42).fit_transform(sparse_matrix)
        return _normalize_embeddings(np.asarray(embeddings, dtype=np.float32)), "tfidf-svd"


def _project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    count = embeddings.shape[0]
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    if count == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)
    if count == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    coords: np.ndarray
    if UMAP is not None:
        try:
            reducer = UMAP(
                n_components=2,
                n_neighbors=max(3, min(20, count - 1)),
                min_dist=0.12,
                metric="cosine",
                random_state=42,
            )
            coords = reducer.fit_transform(embeddings)
        except Exception:
            coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    coords = np.asarray(coords, dtype=np.float32)
    coords -= np.mean(coords, axis=0, keepdims=True)
    scale = float(np.max(np.abs(coords)))
    if scale == 0:
        scale = 1.0
    return (coords / scale) * 100.0


def _choose_cluster_count(total: int) -> int:
    if total <= 1:
        return 1
    heuristic = int(round(math.log2(total)))
    return max(2, min(total, min(12, max(4, heuristic))))


def _format_cluster_label(terms: list[str], fallback_index: int) -> str:
    cleaned = []
    for term in terms:
        term = " ".join(term.split()).strip()
        if not term:
            continue
        cleaned.append(term.title())
    if not cleaned:
        return f"Cluster {fallback_index + 1}"
    if len(cleaned) == 1:
        return cleaned[0]
    return " / ".join(cleaned[:2])


def _summarize_cluster_titles(cluster_titles: list[str], fallback_index: int) -> str:
    if not cluster_titles:
        return f"Cluster {fallback_index + 1}"

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=512)
        matrix = vectorizer.fit_transform(cluster_titles)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        vocab = np.asarray(vectorizer.get_feature_names_out())
        ranked_terms = [vocab[index] for index in scores.argsort()[::-1] if scores[index] > 0]
        return _format_cluster_label(ranked_terms[:4], fallback_index)
    except Exception:
        return f"Cluster {fallback_index + 1}"


def _cluster_entries(entries: list[dict[str, Any]], embeddings: np.ndarray) -> list[dict[str, Any]]:
    total = len(entries)
    if total == 0:
        return []

    cluster_count = _choose_cluster_count(total)
    if cluster_count == 1:
        return [
            {
                "id": 0,
                "name": "All Papers",
                "count": total,
            }
        ]

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        batch_size=min(1024, max(256, total // 2)),
        n_init=10,
    )
    labels = model.fit_predict(embeddings)

    grouped_titles: dict[int, list[str]] = {}
    for entry, label in zip(entries, labels):
        entry["cluster_id"] = int(label)
        grouped_titles.setdefault(int(label), []).append(entry["display_title"])

    cluster_rows = []
    for cluster_id, titles in grouped_titles.items():
        cluster_rows.append(
            {
                "id": cluster_id,
                "name": _summarize_cluster_titles(titles, cluster_id),
                "count": len(titles),
            }
        )

    cluster_rows.sort(key=lambda row: (-row["count"], row["name"].lower()))
    cluster_order = {row["id"]: index for index, row in enumerate(cluster_rows)}
    color_map = {row["id"]: PALETTE[index % len(PALETTE)] for index, row in enumerate(cluster_rows)}
    name_map = {row["id"]: row["name"] for row in cluster_rows}

    for entry in entries:
        cluster_id = entry["cluster_id"]
        entry["cluster_rank"] = cluster_order[cluster_id]
        entry["cluster_name"] = name_map[cluster_id]
        entry["color"] = color_map[cluster_id]

    for row in cluster_rows:
        row["color"] = color_map[row["id"]]

    return cluster_rows


def _build_payload(
    entries: list[dict[str, Any]],
    signature: str,
    embedding_model: str,
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "stats": {
            "total_papers": len(entries),
            "total_clusters": len(clusters),
            "embedding_model": embedding_model,
            "layout": "title-embedding-2d",
        },
        "clusters": clusters,
        "points": entries,
    }


def _empty_payload(message: str) -> dict[str, Any]:
    return {
        "status": "empty",
        "message": message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_papers": 0,
            "total_clusters": 0,
            "embedding_model": "",
            "layout": "title-embedding-2d",
        },
        "clusters": [],
        "points": [],
    }


def _read_cached_payload(expected_signature: str) -> dict[str, Any] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != expected_signature:
        return None
    return payload


def _write_cache(payload: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_library_nebula_payload(force_rebuild: bool = False, json_root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = _resolve_json_root(json_root)
    entries, json_files = _load_library_entries(resolved_root)
    if not resolved_root.exists():
        return _empty_payload(f"Library JSON directory not found: {resolved_root}")
    if not json_files:
        return _empty_payload(f"No library JSON files found in: {resolved_root}")
    if not entries:
        return _empty_payload("Library JSON files were found, but no valid title-to-PDF mappings were loaded.")

    signature = _compute_signature(json_files)
    if not force_rebuild:
        cached = _read_cached_payload(signature)
        if cached is not None:
            return cached

    titles = [entry["title"] for entry in entries]
    embeddings, embedding_model = _embed_titles(titles)
    coords = _project_embeddings(embeddings)
    clusters = _cluster_entries(entries, embeddings)

    cluster_counts = Counter(entry["cluster_id"] for entry in entries)
    for index, entry in enumerate(entries):
        cluster_size = cluster_counts[entry["cluster_id"]]
        entry["x"] = round(float(coords[index, 0]), 4)
        entry["y"] = round(float(coords[index, 1]), 4)
        entry["symbol_size"] = max(6, min(14, 6 + int(math.log2(cluster_size + 1))))

    payload = _build_payload(entries, signature, embedding_model, clusters)
    _write_cache(payload)
    return payload
