from governed_rag import answer_with_governance

print("=" * 80)
print("只检索未过期资料")
fresh = answer_with_governance(
    "原粹树脂应该优先做什么？",
    role="student",
    include_expired=False,
)
print(fresh["answer"])
print("warnings:", fresh["warnings"])

print("=" * 80)
print("包含过期资料，用于观察冲突提示")
with_old = answer_with_governance(
    "原粹树脂应该优先做什么？",
    role="student",
    include_expired=True,
)
print(with_old["answer"])
print("warnings:", with_old["warnings"])
