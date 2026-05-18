import json
import os
import urllib.error
import urllib.request
from chroma_vector_db import search_chroma_db
from governed_rag import detect_conflicts, format_citation

SYSTEM_PROMPT = """你是一个严格基于资料回答的 RAG 助手。
你只能依据 <context> 中的资料回答。
如果资料不足，请明确说"资料中没有足够依据"。
回答必须保留方括号中的引用编号，例如 [genshin_basics_v1#c002]。
如果系统提示存在过期或冲突资料，不要给出唯一确定结论。
回答应简洁，最多 5 条要点，每条不要超过 60 字。"""

def build_context(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        meta = chunk["metadata"]
        status = "已过期" if chunk["expired"] else "有效"
        lines.append(f"[{chunk['chunk_id']}] 标题：{meta['title']}；"
                     f"来源：{meta['source']}；更新时间：{meta['updated_at']}；"
                     f"状态：{status}；内容：{chunk['text']}")
    return "\n".join(lines)

def build_warnings(chunks: list[dict]) -> list[str]:
    warnings = []
    if any(chunk["expired"] for chunk in chunks):
        warnings.append("检索结果包含过期资料，请优先参考未过期来源。")
    conflicts = detect_conflicts(chunks)
    if conflicts:
        warnings.append(f"检测到冲突内容：{conflicts}，需要人工确认。")
    return warnings

def qwen_config() -> dict:
    return {
      "base_url": "http://127.0.0.1:11434",
        "model": os.getenv("QWEN_MODEL", "qwen3.6:7b"),
        "max_tokens": 1024,
    }

def ollama_chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/api/chat"

def call_qwen(messages: list[dict]) -> str:
    config = qwen_config()
    payload = {
        "model": config["model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": config["max_tokens"],
        },
    }
    request = urllib.request.Request(
        ollama_chat_url(config["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Ollama：{exc}") from exc
    message = data.get("message", {})
    answer = (message.get("content") or "").strip()
    if answer:
        return answer
    raise RuntimeError("Qwen 返回了空答案，请检查模型状态。")

def answer_with_qwen(question: str, role: str = "student", include_expired: bool = False) -> dict:
    chunks = search_chroma_db(question=question, role=role, top_k=5, include_expired=include_expired)
    if not chunks:
        return {
            "answer": "资料中没有找到足够依据，无法回答该问题。",
            "citations": [],
            "warnings": [],
            "grounded": False,
        }
    warnings = build_warnings(chunks)
    context = build_context(chunks)
    user_prompt = (f"<context>\n{context}\n</context>\n\n"
                   f"系统提示：{'；'.join(warnings) if warnings else '无'}\n"
                   f"用户问题：{question}")
    answer = call_qwen(messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return {
        "answer": answer,
        "citations": [format_citation(chunk) for chunk in chunks],
        "warnings": warnings,
        "grounded": True,
    }

if __name__ == "__main__":
    # 测试
    result = answer_with_qwen("原石有什么用途？", role="student")
    print("答案:", result["answer"])
    print("引用:", result["citations"][0]["chunk_id"] if result["citations"] else "无")
    print("状态:", "有依据" if result["grounded"] else "拒答")
