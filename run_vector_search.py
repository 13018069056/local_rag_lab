from vector_db import search_vector_db

for role in ["student", "teacher"]:
    print("=" * 80)
    print("role:", role)
    for item in search_vector_db("课堂讲角色养成时有什么建议？", role=role):
        print(item["chunk_id"], item["metadata"]["title"], item["score"])
