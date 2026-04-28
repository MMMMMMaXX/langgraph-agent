# Knowledge Ingestion

`app/knowledge` 是知识库的真相源和导入层，负责把外部文档变成可检索的
chunk，并同步写入两套索引：

- SQLite catalog + FTS5：保存文档/chunk 元数据，服务 lexical retrieval。
- Chroma docs collection：保存 chunk embedding，服务 dense retrieval。

## MVP 导入流程

```text
POST /knowledge/import
-> normalize_import_input
-> chunk_document_text
-> KnowledgeCatalog.upsert_document
-> KnowledgeCatalog.replace_chunks
-> ChromaVectorStore.delete(where={"doc_id": doc_id})
-> embed_texts
-> ChromaVectorStore.upsert
```

导入时会先按 `doc_id` 删除旧 Chroma chunk，再写入新 chunk，避免重导入后
旧向量残留。

## API 示例

### JSON 内容导入

```bash
curl -X POST http://127.0.0.1:8000/knowledge/import \
  -H "Content-Type: application/json" \
  -d '{
    "title": "无障碍指南",
    "source": "accessibility.md",
    "source_type": "md",
    "content": "# WAI-ARIA\n\nWAI-ARIA 是无障碍技术规范，可以帮助屏幕阅读器识别页面状态。"
  }'
```

返回：

```json
{
  "doc_id": "doc-...",
  "title": "无障碍指南",
  "source": "accessibility.md",
  "source_type": "md",
  "content_hash": "...",
  "chunk_count": 1,
  "indexed_to_sqlite": true,
  "indexed_to_chroma": true
}
```

### 文件上传导入

```bash
curl -X POST http://127.0.0.1:8000/knowledge/import/file \
  -F "file=@./docs/accessibility.md" \
  -F "title=无障碍指南" \
  -F 'metadata_json={"topic":"a11y"}'
```

`title`、`doc_id`、`source`、`source_type`、`metadata_json` 都是可选字段。
如果不传 `source_type`，后端会根据文件扩展名推断，例如 `.md`、`.txt`、
`.json`。

查看文档：

```bash
curl http://127.0.0.1:8000/knowledge/docs
curl http://127.0.0.1:8000/knowledge/docs/<doc_id>
```

## 当前边界

- 支持 JSON body 导入，也支持 multipart 文件上传。
- `source_type` 支持 `txt`、`md`、`json`。
- JSON 文档会尝试解析 `id/doc_id`、`title/name`、`source/path`、
  `content/text/body`、`metadata`。
- 文件上传当前按 UTF-8 文本文档读取，不处理 PDF / Word 这类二进制格式。
