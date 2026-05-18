from chroma_vector_db import build_chroma_store, search_chroma_db
from governed_rag import detect_conflicts, format_citation, synthesize_answer

def answer_with_chroma(question: str, role: str = "student", include_expired: bool = False) -> dict:
    chunks = search_chroma_db(question=question, role=role, top_k=5, include_expired=include_expired)
    if not chunks:
        return {"answer": "资料中没有找到足够依据，无法回答该问题。", "citations": [], "warnings": [], "grounded": False}
    warnings = []
    if any(chunk["expired"] for chunk in chunks):
        warnings.append("检索结果包含过期资料，请优先参考未过期来源。")
    conflicts = detect_conflicts(chunks)
    if conflicts:
        warnings.append(f"检测到冲突内容：{conflicts}，需要人工确认。")
    return {"answer": synthesize_answer(question, chunks, warnings),
            "citations": [format_citation(chunk) for chunk in chunks],
            "warnings": warnings, "grounded": True}

build_chroma_store()
CASES = [("课堂讲角色养成时有什么建议？", "student", False),
         ("课堂讲角色养成时有什么建议？", "teacher", False),
         ("原粹树脂应该优先做什么？", "student", True)]

for question, role, include_expired in CASES:
    print("=" * 80)
    print("question:", question, "role:", role)
    result = answer_with_chroma(question, role=role, include_expired=include_expired)
    print(result["answer"])
    print("citations:", result["citations"][:2] if result["citations"] else [])
    print("warnings:", result["warnings"])
