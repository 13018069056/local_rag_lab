import json
from pathlib import Path
from chunker import load_text, split_by_lines
from student_info import print_student_id

print_student_id()

text = load_text("docs/genshin_notes.txt")
chunks = split_by_lines(text)
Path("index").mkdir(exist_ok=True)
Path("index/chunks.json").write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"chunks={len(chunks)}")
