"""轻量文本相似度。

不引入分词依赖：中文按字符二元组（bigram）比较，对短句和关键词检索够用。
所有函数都是纯函数，方便测试与复用（查重、唤醒、主动消息去重）。
"""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_MAX_TEXT = 4000


def normalize(text: str) -> str:
    return _NON_WORD.sub("", (text or "").lower())[:_MAX_TEXT]


def bigrams(text: str) -> set[str]:
    normalized = normalize(text)
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def jaccard(a: str, b: str) -> float:
    """对称相似度，适合判断"是不是同一个东西"。"""
    left, right = bigrams(a), bigrams(b)
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def overlap(a: str, b: str) -> float:
    """包含度，适合"长文里是否提到了短查询"。"""
    left, right = bigrams(a), bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def similarity(a: str, b: str) -> float:
    return max(jaccard(a, b), overlap(a, b) * 0.85)


__all__ = ["bigrams", "jaccard", "normalize", "overlap", "similarity"]