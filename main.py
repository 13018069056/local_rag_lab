from rag_answer import answer_with_context
from student_info import print_student_id

QUESTIONS = [
    "原神玩家扮演谁？",
    "祈愿系统可以抽取什么角色或武器？",
    "元素反应有哪些例子？",
    "今天食堂有什么菜？",
]

print_student_id()
for question in QUESTIONS:
    print("=" * 80)
    print("Q:", question)
    result = answer_with_context(question)
    print(result["answer"])
    print("sources:", result["sources"], "grounded:", result["grounded"])
