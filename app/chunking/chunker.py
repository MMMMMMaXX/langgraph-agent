from app.chunking.models import DocumentChunk
from app.config import CHUNKING_CONFIG

MARKDOWN_HEADING_PREFIXES = ("#", "##", "###", "####", "#####", "######")
SENTENCE_BOUNDARY_CHARS = "。！？!?；;\n"
OVERLAP_START_BOUNDARY_CHARS = "。！？!?；;，,、：: \n\t"
OVERLAP_START_ALIGN_SCAN_CHARS = 40

# 自适应 chunk 参数是经验初始值，用来在默认 280/60 的基础上按文档结构微调。
# 后续应结合 retrieval eval / chunk inspector 指标继续校准。
LONG_DOCUMENT_CHARS = 6000
LONG_TEXT_CHUNK_SIZE_CHARS = 360
LONG_TEXT_CHUNK_OVERLAP_CHARS = 80
STRUCTURED_CHUNK_SIZE_CHARS = 240
STRUCTURED_CHUNK_OVERLAP_CHARS = 40
DEEP_HEADING_CHUNK_OVERLAP_CHARS = 50
FAQ_CHUNK_SIZE_CHARS = 220
FAQ_CHUNK_OVERLAP_CHARS = 40
STEP_CHUNK_SIZE_CHARS = 260
STEP_CHUNK_OVERLAP_CHARS = 50
FAQ_SIGNAL_PREFIXES = ("q:", "q：", "问：", "问题：")
STEP_SIGNAL_PREFIXES = ("- ", "* ", "1. ", "2. ", "3. ", "1、", "2、", "3、")


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    return any(
        stripped.startswith(f"{prefix} ") for prefix in MARKDOWN_HEADING_PREFIXES
    )


def _heading_title(line: str) -> str:
    return line.strip().lstrip("#").strip()


def _heading_level(line: str) -> int:
    stripped = line.strip()
    return len(stripped) - len(stripped.lstrip("#"))


def _split_paragraph_ranges(text: str) -> list[tuple[int, int, str]]:
    """按非空段落切分，并保留段落在规范化文本里的字符区间。"""

    ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for paragraph in text.split("\n\n"):
        start = text.find(paragraph, cursor)
        if start < 0:
            start = cursor
        end = start + len(paragraph)
        content = paragraph.strip()
        if content:
            ranges.append((start, end, content))
        cursor = end + 2
    return ranges


def _split_section_ranges(text: str) -> list[tuple[int, int, str, int]]:
    """按 Markdown 标题拆 section；没有标题时退化为整篇文档。

    工业 RAG 里 chunk 最怕把标题和正文关系切散。这里先做轻量 section 感知：
    看到 Markdown 标题就开启新 section，后续 chunk 会尽量在 section 内打包。
    """

    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, int, str, int]] = []
    section_start = 0
    section_title = ""
    section_level = 0
    offset = 0

    for line in lines:
        if _is_heading_line(line):
            if offset > section_start:
                sections.append((section_start, offset, section_title, section_level))
            section_start = offset
            section_title = _heading_title(line)
            section_level = _heading_level(line)
        offset += len(line)

    if text:
        sections.append((section_start, len(text), section_title, section_level))
    return sections or [(0, len(text), "", 0)]


def _clamp_chunk_params(size: int, overlap: int, minimum: int) -> tuple[int, int, int]:
    overlap = min(overlap, max(size - 1, 0))
    minimum = min(minimum, size)
    return size, overlap, minimum


def _count_prefixed_lines(text: str, prefixes: tuple[str, ...]) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(prefixes):
            count += 1
    return count


def _looks_like_faq_section(section_text: str) -> bool:
    return _count_prefixed_lines(section_text, FAQ_SIGNAL_PREFIXES) >= 2


def _looks_like_step_section(section_text: str) -> bool:
    return _count_prefixed_lines(section_text, STEP_SIGNAL_PREFIXES) >= 3


def _resolve_section_chunk_params(
    *,
    source_type: str,
    document_chars: int,
    section_text: str,
    heading_level: int,
    chunk_size_chars: int | None,
    chunk_overlap_chars: int | None,
    min_chunk_chars: int | None,
) -> tuple[int, int, int]:
    """按文档类型、标题层级和段落形态动态生成 section 级切块参数。"""

    size = chunk_size_chars or CHUNKING_CONFIG.chunk_size_chars
    overlap = chunk_overlap_chars or CHUNKING_CONFIG.chunk_overlap_chars
    minimum = min_chunk_chars or CHUNKING_CONFIG.min_chunk_chars

    if chunk_size_chars is not None:
        return _clamp_chunk_params(size, overlap, minimum)

    normalized_source_type = source_type.strip().lower()
    if normalized_source_type == "json":
        size = min(size, STRUCTURED_CHUNK_SIZE_CHARS)
        overlap = min(overlap, STRUCTURED_CHUNK_OVERLAP_CHARS)
    elif normalized_source_type == "txt" and document_chars >= LONG_DOCUMENT_CHARS:
        size = max(size, LONG_TEXT_CHUNK_SIZE_CHARS)
        overlap = max(overlap, LONG_TEXT_CHUNK_OVERLAP_CHARS)

    if heading_level in {1, 2} and len(section_text) >= size * 2:
        size = max(size, LONG_TEXT_CHUNK_SIZE_CHARS)
        overlap = max(overlap, LONG_TEXT_CHUNK_OVERLAP_CHARS)
    elif heading_level >= 3:
        size = min(size, STRUCTURED_CHUNK_SIZE_CHARS)
        overlap = min(overlap, DEEP_HEADING_CHUNK_OVERLAP_CHARS)

    if _looks_like_faq_section(section_text):
        size = min(size, FAQ_CHUNK_SIZE_CHARS)
        overlap = min(overlap, FAQ_CHUNK_OVERLAP_CHARS)
    elif _looks_like_step_section(section_text):
        size = min(size, STEP_CHUNK_SIZE_CHARS)
        overlap = min(overlap, STEP_CHUNK_OVERLAP_CHARS)

    return _clamp_chunk_params(size, overlap, minimum)


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
    section_title: str = "",
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
        section_title=section_title,
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


def _chunk_text_window(
    doc_id: str,
    text: str,
    *,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    min_chunk_chars: int | None = None,
    chunk_index_start: int = 0,
    offset_base: int = 0,
    section_title: str = "",
) -> list[DocumentChunk]:
    """按“句子边界优先 + 字符窗口兜底”把单篇文档切成多个 chunk。

    当前项目的数据规模还不大，这里优先选择稳定、直观、便于调试的实现：
    1. 文档内容先做轻量规范化
    2. 在窗口内尽量落到句号、问号、分号、换行等自然边界
    3. 保留 overlap，减少信息恰好落在边界时的召回损失

    这比纯固定窗口更不容易切断语义，同时仍保留简单可控的字符长度上限。
    """

    if not text:
        return []

    size = chunk_size_chars or CHUNKING_CONFIG.chunk_size_chars
    overlap = chunk_overlap_chars or CHUNKING_CONFIG.chunk_overlap_chars
    minimum = min_chunk_chars or CHUNKING_CONFIG.min_chunk_chars

    # overlap 必须小于 size，否则窗口不会前进。
    overlap = min(overlap, max(size - 1, 0))
    step = max(size - overlap, 1)

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = chunk_index_start
    text_len = len(text)

    while start < text_len:
        max_end = min(start + size, text_len)
        end = _find_boundary(text, start, max_end, minimum)
        chunk_text = text[start:end].strip()

        if chunk_text and len(chunk_text) >= minimum:
            chunks.append(
                _build_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=offset_base + start,
                    end_char=offset_base + end,
                    section_title=section_title,
                )
            )
            chunk_index += 1

        if max_end >= text_len:
            break

        # 下一块从当前边界前回退 overlap，既保留上下文，又保证窗口继续前进。
        raw_next_start = max(end - overlap, start + 1)
        next_start = _align_overlap_start(text, raw_next_start, end)
        start = min(next_start, start + step)

    return chunks


def _pack_section_paragraphs(
    *,
    doc_id: str,
    section_text: str,
    section_offset: int,
    section_title: str,
    chunk_index_start: int,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[DocumentChunk]:
    """在 section 内按段落打包，段落过长时再退化到窗口切分。"""

    chunks: list[DocumentChunk] = []
    pending_parts: list[str] = []
    pending_start: int | None = None
    pending_end = 0
    chunk_index = chunk_index_start

    def flush_pending() -> None:
        nonlocal pending_parts, pending_start, pending_end, chunk_index
        if pending_start is None:
            return
        content = "\n\n".join(part.strip() for part in pending_parts if part.strip())
        if len(content) >= min_chunk_chars:
            chunks.append(
                _build_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=content,
                    start_char=section_offset + pending_start,
                    end_char=section_offset + pending_end,
                    section_title=section_title,
                )
            )
            chunk_index += 1
        pending_parts = []
        pending_start = None
        pending_end = 0

    for start, end, paragraph in _split_paragraph_ranges(section_text):
        if len(paragraph) > chunk_size_chars:
            flush_pending()
            long_chunks = _chunk_text_window(
                doc_id,
                paragraph,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
                min_chunk_chars=min_chunk_chars,
                chunk_index_start=chunk_index,
                offset_base=section_offset + start,
                section_title=section_title,
            )
            chunks.extend(long_chunks)
            chunk_index += len(long_chunks)
            continue

        pending_len = len("\n\n".join(pending_parts + [paragraph]))
        if pending_parts and pending_len > chunk_size_chars:
            flush_pending()

        if pending_start is None:
            pending_start = start
        pending_parts.append(paragraph)
        pending_end = end

    flush_pending()
    return chunks


def chunk_document_text(
    doc_id: str,
    text: str,
    *,
    chunk_size_chars: int | None = None,
    chunk_overlap_chars: int | None = None,
    min_chunk_chars: int | None = None,
    source_type: str = "",
) -> list[DocumentChunk]:
    """按“标题/段落优先 + 字符窗口兜底”把单篇文档切成多个 chunk。

    新版切片先保留文档自然结构：Markdown 标题形成 section，段落在 section
    内打包；只有单段过长时才用字符窗口切分。这样更适合后续引用展示和
    FTS/BM25 检索，也减少标题、定义、解释被切散的概率。
    """

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    for (
        section_start,
        section_end,
        section_title,
        heading_level,
    ) in _split_section_ranges(normalized):
        section_text = normalized[section_start:section_end].strip()
        if not section_text:
            continue
        size, overlap, minimum = _resolve_section_chunk_params(
            source_type=source_type,
            document_chars=len(normalized),
            section_text=section_text,
            heading_level=heading_level,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
        section_chunks = _pack_section_paragraphs(
            doc_id=doc_id,
            section_text=section_text,
            section_offset=section_start,
            section_title=section_title,
            chunk_index_start=len(chunks),
            chunk_size_chars=size,
            chunk_overlap_chars=overlap,
            min_chunk_chars=minimum,
        )
        chunks.extend(section_chunks)

    # 极短文本会因为 minimum 被过滤掉；这种情况下退化成单块，避免文档丢失。
    if not chunks:
        chunks.append(
            _build_chunk(
                doc_id=doc_id,
                chunk_index=0,
                text=normalized,
                start_char=0,
                end_char=len(normalized),
            )
        )

    return chunks
