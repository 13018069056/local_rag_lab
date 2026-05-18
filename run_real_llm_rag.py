from chroma_vector_db import build_chroma_store
from llm_rag_answer import answer_with_qwen
from student_info import print_student_id

print_student_id()
print("正在构建 ChromaDB 知识库...")
build_chroma_store()

QUESTIONS = [
    "原石有什么用途？",
    "课堂讲角色养成时有什么建议？",
    "原粹树脂应该优先做什么？",
]

print("\n" + "="*60)
print("真实大模型 RAG 问答测试")
print("="*60)

for question in QUESTIONS:
    print(f"\n【问题】: {question}")
    try:
        result = answer_with_qwen(question, role="student", include_expired=False)
        print(f"【回答】: {result['answer']}")
        print(f"【引用】: {[c['chunk_id'] for c in result['citations'][:2]]}")
        print(f"【警告】: {result['warnings']}")
        print(f"【状态】: {'有依据' if result['grounded'] else '拒答'}")
    except Exception as e:
        print(f"【错误】: {e}")
        print("请检查 Ollama 是否运行，模型是否已下载")

print("\n" + "="*60)
print("包含过期资料测试（观察冲突提示）")
print("="*60)

try:
    result = answer_with_qwen("原粹树脂应该优先做什么？", role="student", include_expired=True)
    print(f"【回答】: {result['answer']}")
    print(f"【警告】: {result['warnings']}")
except Exception as e:
    print(f"【错误】: {e}")
