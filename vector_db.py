import json
from datetime import date
from pathlib import Path
from simple_vector import cosine, vectorize

def parse_date(value: str | None):
    if not value:
        return None
    return date.fromisoformat(value)

def is_expired(metadata: dict, today: date | None = None) -> bool:
    today = today or date.today()
    expires_at = parse_date(metadata.get("expires_at"))
    return bool(expires_at and expires_at < today)

def can_access(metadata: dict, role: str) -> bool:
    return role in metadata.get("roles", [])

def build_vector_store(corpus_path: str = "docs/genshin_corpus.json",
                       output_path: str = "index/vector_store.json") -> None:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    records = []
    for doc in corpus["documents"]:
        for index, text in enumerate(doc["content"], start=1):
            chunk_id = f"{doc['doc_id']}#c{index:03d}"
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "updated_at": doc["updated_at"],
                "expires_at": doc["expires_at"],
                "roles": doc["roles"],
                "claim_key": doc.get("claim_key"),
                "claim_value": doc.get("claim_value"),
            }
            records.append({
                "chunk_id": chunk_id,
                "text": text,
                "vector": dict(vectorize(text)),
                "metadata": metadata,
            })
    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"vector_records={len(records)}")

def load_vector_store(path: str = "index/vector_store.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def search_vector_db(question: str, role: str = "student", top_k: int = 5,
                     include_expired: bool = False) -> list[dict]:
    query_vec = vectorize(question)
    scored = []
    for record in load_vector_store():
        metadata = record["metadata"]
        if not can_access(metadata, role):
            continue
        if is_expired(metadata) and not include_expired:
            continue
        score = cosine(query_vec, record["vector"])
        if score <= 0:
            continue
        item = {**record, "score": round(score, 4), "expired": is_expired(metadata)}
        scored.append(item)
    scored.sort(key=lambda x: (x["expired"], -x["score"]))
    return scored[:top_k]

if __name__ == "__main__":
    build_vector_store()
