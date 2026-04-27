"""Lexical retrieval tokenizer helpers.

FTS5 默认 unicode61 tokenizer 对中文没有业务级分词能力，所以我们在写索引和
构造 query 时显式加入 jieba 分词结果与中文 bigram。原始 chunk 内容仍保留在
document_chunks.content，回答和引用不会使用增强后的索引文本。
"""

from __future__ import annotations

import re

FTS_SPECIAL_CHARS = re.compile(r'["]')
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
MIN_SEGMENT_TERM_CHARS = 2

QUERY_STOPWORDS = {
    "是什么",
    "什么是",
    "为什么",
    "怎么",
    "如何",
    "一下",
    "这个",
    "那个",
}


def _escape_fts_phrase(value: str) -> str:
    return FTS_SPECIAL_CHARS.sub(" ", value).strip()


def _dedupe_terms(terms: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = _escape_fts_phrase(term)
        if not cleaned or cleaned in QUERY_STOPWORDS or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def cjk_bigrams(text: str) -> list[str]:
    chars = CJK_CHAR_RE.findall(text)
    return [
        "".join(chars[index : index + 2])
        for index in range(0, max(len(chars) - 1, 0))
    ]


def jieba_terms(text: str) -> list[str]:
    """返回 jieba 中文分词结果；未安装时退回空列表。

    requirements.txt 已声明 jieba，这里的兜底只是为了让单元测试和极简环境更稳。
    """

    try:
        import jieba
    except ImportError:
        return []

    terms = []
    for term in jieba.cut(text, cut_all=False):
        cleaned = term.strip()
        if len(cleaned) < MIN_SEGMENT_TERM_CHARS:
            continue
        if ASCII_TOKEN_RE.fullmatch(cleaned):
            continue
        terms.append(cleaned)
    return terms


def lexical_terms(text: str) -> list[str]:
    """提取用于 lexical query/index 的稳定词项。"""

    terms: list[str] = []
    terms.extend(ASCII_TOKEN_RE.findall(text))
    terms.extend(jieba_terms(text))
    terms.extend(cjk_bigrams(text))

    cjk_chars = CJK_CHAR_RE.findall(text)
    if not terms and cjk_chars:
        terms.extend(cjk_chars)

    return _dedupe_terms(terms)


def build_fts_query(query: str) -> str:
    """把自然语言 query 转成 FTS5 MATCH 表达式。"""

    terms = lexical_terms(query.strip())
    return " OR ".join(f'"{term}"' for term in terms)


def build_fts_index_text(text: str) -> str:
    """为 FTS5 构造增强索引文本。"""

    terms = lexical_terms(text)
    return f"{text}\n{' '.join(terms)}".strip()
