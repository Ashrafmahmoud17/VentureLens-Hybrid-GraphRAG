"""Hybrid retrieval for SkillBridge: Chroma vector + Neo4j full-text + graph context.

Run from the project root:
    python -m src.hybrid_retrieval --setup-indexes
    python -m src.hybrid_retrieval --question "What skills are needed for a Data Scientist?"

The existing local SentenceTransformer model and Chroma collection are reused.
"""

from __future__ import annotations
import shutil
import argparse
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Support both names used in the original project while standardizing on NEO4J_URI.
NEO4J_URI = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

RAILWAY_VOLUME_MOUNT_PATH = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")

if RAILWAY_VOLUME_MOUNT_PATH:
    DEFAULT_CHROMA_PATH = Path(RAILWAY_VOLUME_MOUNT_PATH) / "chroma_db"
else:
    DEFAULT_CHROMA_PATH = BASE_DIR / "chroma_db"

CHROMA_PATH = Path(
    os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_PATH))
)
COLLECTION_NAME = "skillbridge_embeddings"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Each index only includes properties that actually exist on that node label.
FULLTEXT_INDEXES = {
    "skillbridge_skill_fulltext": ("Skill", "skill_name", "category", "description"),
    "skillbridge_course_fulltext": ("Course", "course_name", "difficulty", "description"),
    "skillbridge_job_fulltext": ("Job", "job_title", "description"),
    "skillbridge_project_fulltext": ("Project", "project_name", "difficulty", "description"),
}

_model = None
_chroma_collection = None


def _require_neo4j_config() -> None:
    missing = [
        name for name, value in {
            "NEO4J_URI (or NEO4J_URL)": NEO4J_URI,
            "NEO4J_USERNAME": NEO4J_USERNAME,
            "NEO4J_PASSWORD": NEO4J_PASSWORD,
        }.items() if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Neo4j settings in the project-root .env: " + ", ".join(missing)
        )


def _get_driver():
    _require_neo4j_config()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def _get_collection():
    global _chroma_collection

    if _chroma_collection is None:
        import chromadb

        # Original database committed to the repository.
        source_path = BASE_DIR / "chroma_db"

        # On Railway, seed the persistent volume on first startup.
        if RAILWAY_VOLUME_MOUNT_PATH:
            database_file = CHROMA_PATH / "chroma.sqlite3"

            if not database_file.exists():
                if not (source_path / "chroma.sqlite3").exists():
                    raise RuntimeError(
                        f"Original Chroma database was not found at {source_path}"
                    )

                CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)

                shutil.copytree(
                    source_path,
                    CHROMA_PATH,
                    dirs_exist_ok=True,
                )

                print(f"Chroma database initialized at {CHROMA_PATH}")

        if not CHROMA_PATH.exists():
            raise RuntimeError(
                f"Chroma database was not found at {CHROMA_PATH}. "
                "Check CHROMA_PATH or the Railway Volume configuration."
            )

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        try:
            _chroma_collection = client.get_collection(
                name=COLLECTION_NAME
            )
        except Exception as exc:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION_NAME}' was not found "
                f"at {CHROMA_PATH}."
            ) from exc

    return _chroma_collection

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def create_fulltext_indexes() -> None:
    """Create the four Neo4j full-text indexes required by keyword retrieval."""
    driver = _get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for index_name, (label, *properties) in FULLTEXT_INDEXES.items():
                property_list = ", ".join(f"n.{prop}" for prop in properties)
                query = f"""
                CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON EACH [{property_list}]
                """
                session.run(query).consume()
                print(f"Ready: {index_name}")
        print("All Hybrid Retrieval full-text indexes are ready.")
    finally:
        driver.close()


def _to_lucene_query(question: str) -> str:
    """Turn user text into a safe OR query for Neo4j's Lucene full-text parser."""
    tokens = re.findall(r"[\w+#.\-]+", question, flags=re.UNICODE)
    if not tokens:
        return ""
    special = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
    escaped = [special.sub(r"\\\1", token) for token in tokens]
    return " OR ".join(escaped)


def _vector_search(question: str, top_k: int) -> list[dict[str, Any]]:
    collection = _get_collection()
    embedding = _get_model().encode(question, convert_to_numpy=True)
    result = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=max(1, top_k),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadata = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for rank, node_id in enumerate(ids, start=1):
        meta = metadata[rank - 1] or {}
        hits.append({
            "node_id": str(node_id),
            "node_type": meta.get("node_type"),
            "document": docs[rank - 1] if rank - 1 < len(docs) else "",
            "distance": distances[rank - 1] if rank - 1 < len(distances) else None,
            "rank": rank,
            "source": "vector",
        })
    return hits


def _keyword_search(question: str, top_k: int) -> list[dict[str, Any]]:
    lucene_query = _to_lucene_query(question)
    if not lucene_query:
        return []

    driver = _get_driver()
    hits: list[dict[str, Any]] = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for index_name, (label, *_properties) in FULLTEXT_INDEXES.items():
                query = """
                CALL db.index.fulltext.queryNodes($index_name, $query)
                YIELD node, score
                RETURN $label AS node_type,
                       CASE $label
                           WHEN 'Skill' THEN node.skill_id
                           WHEN 'Course' THEN node.course_id
                           WHEN 'Job' THEN node.job_id
                           WHEN 'Project' THEN node.project_id
                       END AS node_id,
                       properties(node) AS properties,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """
                rows = session.run(
                    query,
                    parameters={
                        "index_name": index_name,
                        "query": lucene_query,
                        "label": label,
                        "top_k": max(1, top_k),
                    },
                )
                for row in rows:
                    if row["node_id"] is not None:
                        hits.append({
                            "node_id": str(row["node_id"]),
                            "node_type": row["node_type"],
                            "properties": row["properties"],
                            "keyword_score": float(row["score"]),
                            "source": "keyword",
                        })
    except Exception as exc:
        message = str(exc)
        if "not found" in message.lower() or "no such" in message.lower():
            raise RuntimeError(
                "Neo4j full-text indexes are missing. Run: "
                "python -m src.hybrid_retrieval --setup-indexes"
            ) from exc
        raise
    finally:
        driver.close()

    # Merge duplicate matches across indexes and rank by Neo4j full-text score.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for hit in hits:
        key = (hit["node_type"], hit["node_id"])
        if key not in unique or hit["keyword_score"] > unique[key]["keyword_score"]:
            unique[key] = hit
    ordered = sorted(unique.values(), key=lambda item: item["keyword_score"], reverse=True)
    for rank, hit in enumerate(ordered, start=1):
        hit["rank"] = rank
    return ordered[: max(1, top_k)]


def _rrf_fusion(
    vector_hits: list[dict[str, Any]],
    keyword_hits: list[dict[str, Any]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion; score = sum(1 / (rrf_k + rank))."""
    fused: dict[tuple[str, str], dict[str, Any]] = {}
    for source_hits, source in ((vector_hits, "vector"), (keyword_hits, "keyword")):
        for hit in source_hits:
            node_type = hit.get("node_type")
            node_id = hit.get("node_id")
            if not node_type or not node_id:
                continue
            key = (str(node_type), str(node_id))
            item = fused.setdefault(key, {
                "node_type": str(node_type),
                "node_id": str(node_id),
                "rrf_score": 0.0,
                "sources": [],
            })
            item["rrf_score"] += 1.0 / (rrf_k + int(hit["rank"]))
            if source not in item["sources"]:
                item["sources"].append(source)
            if source == "vector":
                item["document"] = hit.get("document", "")
                item["vector_distance"] = hit.get("distance")
            else:
                item["properties"] = hit.get("properties", {})
                item["keyword_score"] = hit.get("keyword_score")

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:max(1, top_k)]


def _expand_graph_context(items: list[dict[str, Any]], neighbors_per_node: int = 5) -> list[dict[str, Any]]:
    if not items:
        return items

    driver = _get_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for item in items:
                label = item["node_type"]
                id_property = {
                    "Skill": "skill_id",
                    "Course": "course_id",
                    "Job": "job_id",
                    "Project": "project_id",
                }.get(label)
                if not id_property:
                    continue

                # Label/property names come only from the fixed allow-list above.
                query = f"""
                MATCH (n:{label} {{{id_property}: $node_id}})
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN properties(n) AS node_properties,
                       labels(n) AS node_labels,
                       collect(DISTINCT CASE WHEN m IS NULL THEN null ELSE {{
                           relationship: type(r),
                           direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                           labels: labels(m),
                           properties: properties(m)
                       }} END)[0..$neighbor_limit] AS neighbors
                """
                record = session.run(
                    query,
                    node_id=item["node_id"],
                    neighbor_limit=max(0, neighbors_per_node),
                ).single()
                if record:
                    item["node_properties"] = record["node_properties"]
                    item["node_labels"] = record["node_labels"]
                    item["neighbors"] = [n for n in (record["neighbors"] or []) if n]
                else:
                    item["node_properties"] = item.get("properties", {})
                    item["neighbors"] = []
    finally:
        driver.close()
    return items


def hybrid_search(
    question: str,
    top_k: int = 5,
    vector_k: int = 10,
    keyword_k: int = 10,
    neighbors_per_node: int = 5,
) -> dict[str, Any]:
    """Retrieve relevant nodes using Vector + Keyword + RRF + Graph expansion."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_hits = _vector_search(question.strip(), vector_k)
    keyword_hits = _keyword_search(question.strip(), keyword_k)
    fused = _rrf_fusion(vector_hits, keyword_hits, top_k)
    enriched = _expand_graph_context(fused, neighbors_per_node)

    return {
        "question": question,
        "retrieval_method": "vector + keyword + reciprocal rank fusion + graph expansion",
        "counts": {
            "vector": len(vector_hits),
            "keyword": len(keyword_hits),
            "fused": len(fused),
        },
        "results": enriched,
    }


def _format_context(retrieval: dict[str, Any]) -> str:
    sections = []
    for index, item in enumerate(retrieval["results"], start=1):
        sections.append(
            f"Result {index}\n"
            f"Type: {item['node_type']}\n"
            f"ID: {item['node_id']}\n"
            f"Properties: {item.get('node_properties') or item.get('properties') or {}}\n"
            f"Found by: {', '.join(item.get('sources', []))}\n"
            f"RRF score: {item['rrf_score']:.6f}\n"
            f"Graph neighbors: {item.get('neighbors', [])}"
        )
    return "\n\n".join(sections) if sections else "No relevant graph records were retrieved."


def ask_groq(question: str, top_k: int = 5) -> dict[str, Any]:
    """Run hybrid retrieval and pass the retrieved context to Groq for an answer."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the project-root .env file.")

    from groq import Groq

    retrieval = hybrid_search(question, top_k=top_k)
    context = _format_context(retrieval)
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SkillBridge, an assistant for skills, courses, jobs, and projects. "
                    "Answer using only the retrieved context. If the context is insufficient, "
                    "say what is missing instead of inventing facts. Explain relevant graph relationships."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}",
            },
        ],
        temperature=0.2,
    )
    return {
        "answer": completion.choices[0].message.content,
        "retrieval": retrieval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillBridge Hybrid Retrieval")
    parser.add_argument("--setup-indexes", action="store_true", help="Create Neo4j full-text indexes")
    parser.add_argument("--question", type=str, help="Ask a question using Hybrid Retrieval + Groq")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results (default: 5)")
    parser.add_argument("--retrieve-only", action="store_true", help="Run retrieval without calling Groq")
    args = parser.parse_args()

    if args.setup_indexes:
        create_fulltext_indexes()
        return

    if args.question:
        if args.retrieve_only:
            retrieval = hybrid_search(args.question, top_k=args.top_k)
            print("\nHybrid Retrieval Results\n" + "=" * 60)
        else:
            response = ask_groq(args.question, top_k=args.top_k)
            retrieval = response["retrieval"]
            print("\nAnswer\n" + "=" * 60)
            print(response["answer"])
        print("\nRetrieval counts:", retrieval["counts"])
        print("\nRetrieved nodes:")
        for item in retrieval["results"]:
            print(
                f"- {item['node_type']} {item['node_id']} | "
                f"RRF={item['rrf_score']:.6f} | sources={item['sources']}"
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
