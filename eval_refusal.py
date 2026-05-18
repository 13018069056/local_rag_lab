from rag_answer import answer_with_context

OUT_OF_SCOPE = [
    "今天北京天气怎么样？",
    "请推荐一款手机。",
    "请解释量子力学的测不准原理。",
]

for question in OUT_OF_SCOPE:
    result = answer_with_context(question)
    print(question, result["grounded"], result["answer"])
