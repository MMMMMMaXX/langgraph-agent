from app.chunking.models import DocumentChunk
from app.config import CHUNKING_CONFIG

SENTENCE_BOUNDARY_CHARS = "。！？!?；;\n"
OVERLAP_START_BOUNDARY_CHARS = "。！？!?；;，,、：: \n\t"
OVERLAP_START_ALIGN_SCAN_CHARS = 40


def _normalize_text(text: str) -> str:
    """对原文做轻量清洗。

    当前只做最小必要处理：
    - 去掉首尾空白
    - 合并连续空行

    这里故意不做激进清洗，避免破坏原始文档内容和位置映射。
    """

    lines = [line.rstrip() for line in text.strip().splitlines()]
    cleaned: list[str] = []
    last_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and last_blank:
            continue
        cleaned.append(line)
        last_blank = is_blank

    return "\n".join(cleaned).strip()


def _build_chunk(
    *,
    doc_id: str,
    chunk_index: int,
    text: str,
    start_char: int,
    end_char: int,
) -> DocumentChunk:
    chunk_text = text.strip()
    return DocumentChunk(
        chunk_id=f"{doc_id}::chunk::{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=chunk_text,
        start_char=start_char,
        end_char=end_char,
        char_len=len(chunk_text),
    )


def _find_boundary(text: str, start: int, max_end: int, minimum: int) -> int:
    """在固定窗口内优先寻找句子/段落边界。

    固定字符切块容易把一句话从中间切断，影响 embedding 表达。
    这里在 `start + minimum` 到 `max_end` 之间倒序找中文/英文标点或换行，
    找不到时才退回固定窗口。
    """

    lower_bound = min(start + minimum, max_end)
    for index in range(max_end - 1, lower_bound - 1, -1):
        if text[index] in SENTENCE_BOUNDARY_CHARS:
            return index + 1
    return max_end


def _skip_start_separators(text: str, start: int, max_end: int) -> int:
    """跳过 chunk 起点前的空白和分隔符，让片段开头更自然。"""

    index = start
    while index < max_end and text[index] in OVERLAP_START_BOUNDARY_CHARS:
        index += 1
    return index


def _is_ascii_word_char(char: str) -> bool:
    """判断字符是否属于英文单词/数字，避免 overlap 从单词中间切入。"""

    return char.isascii() and char.isalnum()


def _align_overlap_start(text: str, raw_start: int, max_start: int) -> int:
    """把 overlap 起点向后对齐到更自然的文本边界。

    `raw_start` 通常来自 `end - overlap`。这个位置能保证召回覆盖，
    但可能落在英文单词中间，比如 `VoiceOver` 被切成 `eOver`。
    这里只在很小的窗口内向后微调：
    - 优先找中文/英文标点、空格、换行之后的位置
    - 如果已经落在英文单词中间，就前进到该单词结束
    - 找不到合适边界时保持原起点，避免过度削弱 overlap
    """

    if raw_start <= 0:
        return 0

    text_len = len(text)
    safe_max_start = min(max_start, text_len - 1)
    if raw_start >= safe_max_start:
        return raw_start

    scan_end = min(
        safe_max_start,
        raw_start + OVERLAP_START_ALIGN_SCAN_CHARS,
    )

    for index in range(raw_start, scan_end):
        if text[index] in OVERLAP_START_BOUNDARY_CHARS:
            return _skip_start_separators(text, index + 1, max_start)

    previous_char = text[raw_start - 1]
    current_char = text[raw_start]
    if _is_ascii_word_char(previous_char) and _is_ascii_word_char(current_char):
        index = raw_start
        while index < scan_end and _is_ascii_word_char(text[index]):
            index += 1
        return _skip_start_separators(text, index, max_start)

    return raw_start


def chunk_document_text(
    doc_id: str,
    text: str,
    *,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    min_chunk_chars: int | None = None,
) -> list[DocumentChunk]:
    """按“句子边界优先 + 字符窗口兜底”把单篇文档切成多个 chunk。

    当前项目的数据规模还不大，这里优先选择稳定、直观、便于调试的实现：
    1. 文档内容先做轻量规范化
    2. 在窗口内尽量落到句号、问号、分号、换行等自然边界
    3. 保留 overlap，减少信息恰好落在边界时的召回损失

    这比纯固定窗口更不容易切断语义，同时仍保留简单可控的字符长度上限。
    """

    normalized = _normalize_text(text)
    if not normalized:
        return []

    size = chunk_size_chars or CHUNKING_CONFIG.chunk_size_chars
    overlap = chunk_overlap_chars or CHUNKING_CONFIG.chunk_overlap_chars
    minimum = min_chunk_chars or CHUNKING_CONFIG.min_chunk_chars

    # overlap 必须小于 size，否则窗口不会前进。
    overlap = min(overlap, max(size - 1, 0))
    step = max(size - overlap, 1)

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    text_len = len(normalized)

    while start < text_len:
        max_end = min(start + size, text_len)
        end = _find_boundary(normalized, start, max_end, minimum)
        chunk_text = normalized[start:end].strip()

        if chunk_text and len(chunk_text) >= minimum:
            chunks.append(
                _build_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_index += 1

        if max_end >= text_len:
            break

        # 下一块从当前边界前回退 overlap，既保留上下文，又保证窗口继续前进。
        raw_next_start = max(end - overlap, start + 1)
        next_start = _align_overlap_start(normalized, raw_next_start, end)
        start = min(next_start, start + step)

    # 极短文本会因为 minimum 被过滤掉；这种情况下退化成单块，避免文档丢失。
    if not chunks:
        chunks.append(
            _build_chunk(
                doc_id=doc_id,
                chunk_index=0,
                text=normalized,
                start_char=0,
                end_char=text_len,
            )
        )

    return chunks
