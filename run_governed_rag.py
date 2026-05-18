from governed_rag import answer_with_governance

CASES = [
    ("原石、祈愿和角色养成有什么关系？", "student", False),
    ("课堂讲角色养成时有什么建议？", "student", False),
    ("课堂讲角色养成时有什么建议？", "teacher", False),
]

for question, role, include_expired in CASES:
    print("=" * 80)
    print("question:", question, "role:", role)
    result = answer_with_governance(question, role=role, include_expired=include_expired)
    print(result["answer"])
    print("citations:", result["citations"][:2] if result["citations"] else [])
    print("warnings:", result["warnings"])
