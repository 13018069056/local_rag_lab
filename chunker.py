from pathlib import Path

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def split_by_lines(text: str) -> list[dict]:
    chunks = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        chunks.append({
            "id": f"chunk_{i:03d}",
            "text": line,
            "source": "genshin_notes.txt",
        })
    return chunks
