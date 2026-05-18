import json
from pathlib import Path
from simple_vector import cosine, vectorize

def load_chunks(path: str = "index/chunks.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def retrieve(question: str, top_k: int = 3) -> list[dict]:
    query_vec = vectorize(question)
    scored = []
    for chunk in load_chunks():
        score = cosine(query_vec, vectorize(chunk["text"]))
        scored.append({**chunk, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [item for item in scored[:top_k] if item["score"] > 0]
