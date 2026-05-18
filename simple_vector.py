import math
import re
from collections import Counter

def tokenize(text: str) -> list[str]:
    english = re.findall(r"[A-Za-z][A-Za-z0-9_/-]*", text.lower())
    chinese_terms = []
    for term in ["原神", "玩家", "旅行者", "提瓦特", "祈愿", "原石", "纠缠之缘", 
                 "相遇之缘", "角色", "武器", "养成", "等级", "天赋", "命之座", 
                 "圣遗物", "主属性", "副属性", "攻击", "防御", "生命", "元素", 
                 "反应", "蒸发", "融化", "超载", "感电", "扩散", "原粹树脂", 
                 "体力", "地脉", "秘境", "首领", "奖励", "建议", "优先"]:
        if term.lower() in text.lower():
            chinese_terms.append(term.lower())
    return english + chinese_terms

def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))

def cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
