from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """描述文档切块后的最小检索单元。

    字段设计尽量贴近后续 Chroma 元数据结构，方便直接写库：
    - chunk_id: 全局唯一 chunk 主键
    - doc_id: 原始文档 id
    - chunk_index: 文档内顺序
    - text: 当前 chunk 文本
    - start_char / end_char: 对应原文字符区间，方便调试召回来源
    - char_len: chunk 文本长度，方便评估切块粒度是否合理
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    char_len: int
