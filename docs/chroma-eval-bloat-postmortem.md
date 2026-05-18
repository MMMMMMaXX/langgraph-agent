# Chroma Eval Bloat 事故复盘

## 背景

项目的知识库索引采用 SQLite Catalog / FTS5 作为结构化与 lexical 检索层，Chroma 作为 dense vector 检索层。评估脚本会反复导入 fixture 文档、构建索引、执行检索与回答质量评估。

一次本地 eval 后发现磁盘可用空间快速下降，`data/chroma/<segment>/data_level0.bin` 单目录膨胀到约 250GB，导致 baseline eval 无法继续运行。

## 现象

- `data/chroma` 下某个 HNSW segment 文件异常膨胀，单目录约 250GB。
- 磁盘剩余空间仅约 4.6GB / 460GB。
- `/knowledge/health` 的 chunk count 一度可以恢复为一致，但它只检查数量和 ID，一开始不能发现 segment 文件尺寸异常。
- eval 多次运行后，即使文档内容并未变化，Chroma 仍持续发生 delete / upsert。

## 根因

根因不是“文档真的很多”，而是 **eval 反复导入相同文档时触发 Chroma 的 delete + upsert churn，叠加高维向量和 HNSW segment 不自动压缩，导致持久化文件持续膨胀**。

关键链路：

1. 导入文档时原逻辑无条件执行：

   ```python
   store.delete(collection_name="docs", where={"doc_id": doc_id})
   collection.upsert(...)
   ```

2. eval 每次运行会重复 import 同一批 fixture 文档。

3. 即使文档内容不变，也会重复 delete old chunks + upsert new chunks。

4. Chroma HNSW index 对 delete / upsert 不是原地紧凑替换。删除通常更像逻辑删除或队列操作，底层 segment 文件不会自动 shrink。

5. 当前 embedding 维度较高，单条向量占用较大。HNSW segment 还会按容量扩容，一旦扩容，文件体积不会因为删除旧向量自动回落。

6. 多次 eval 或异常中断后，queue / segment 更容易积累未压缩状态，最终表现为 `data_level0.bin` 暴涨。

## 为什么“文档不变”也会变大

直觉上会以为同一个 `doc_id` 重新 upsert 应该覆盖旧数据，体积不变。但向量库内部通常不是简单的 key-value 原地覆盖：

- delete 可能只是标记旧向量不可见。
- upsert 可能写入新的向量记录。
- HNSW 图索引为了查询性能会预分配或扩容 segment。
- 删除后的空间不一定被立即复用，也不一定自动 compaction。

所以对 Chroma/HNSW 来说，“重复 delete + upsert 相同内容”仍可能造成磁盘增长。

## 线上风险

如果线上导入接口没有幂等保护，用户或任务系统反复导入相同文档，也可能触发类似问题：

- Chroma segment 体积异常增长。
- 磁盘被打满，影响服务启动、写入 checkpoint、写 conversation history。
- health 只看 count 时可能误报 `ok`。
- reindex 可以恢复索引一致性，但不能解决根因。

## 修复方案

### 1. Eval Chroma 隔离与自动清理

eval 不再默认写生产 `data/chroma`，而是使用每次 eval 独立的 Chroma 目录。

运行成功时自动清理隔离目录；运行失败或显式 `--keep-chroma` 时保留现场，方便排查。

manifest 中记录：

- `chroma_persist_dir`
- `chroma_cleaned`
- `chroma_keep_reason`
- `chroma_size_bytes_before_cleanup`
- `chroma_size_bytes_after_cleanup`
- `run_status`
- `pass_rate`

这样 eval 不再污染长期本地索引，也能在失败时保留证据。

### 2. Import Content Hash 幂等

导入前计算 normalized content 的 `content_hash`。

当 catalog 中已有同 `doc_id`，并且以下索引相关字段都一致时，直接跳过重建：

- `content_hash`
- `title`
- `source`
- `source_type`
- `metadata`

返回：

```json
{
  "indexed_to_sqlite": false,
  "indexed_to_chroma": false,
  "skipped_reason": "content_unchanged"
}
```

这样相同文档重复导入不会再触发 Chroma delete / upsert。

如果正文或索引相关 metadata 变化，则仍然执行完整重建，保证 ACL、标签、标题、来源等信息不会陈旧。

### 3. Health Segment Size 告警

后续 health 应补充 Chroma segment 文件体积检查，而不仅仅检查 chunk count：

- 统计 `CHROMA_PERSIST_DIR` 下总大小。
- 统计最大单文件大小。
- 计算平均每 chunk 占用。
- 超过阈值时返回 `chroma_segment_size_anomaly` warning。

建议默认阈值：

- 单 segment 文件 > 1GB 触发 warning。
- 总占用 / chunk_count > 10MB 触发 warning。

## 验证方式

### Eval 隔离清理

```bash
EVAL_CASE_IDS=unknown_concept_fallback \
./.venv/bin/python scripts/run_eval_profile.py --profile baseline
```

检查 manifest：

```bash
cat outputs/eval_runs/<run>.manifest.json
```

预期：

- `run_status = success`
- `chroma_cleaned = true`
- `chroma_size_bytes_before_cleanup > 0`
- `chroma_size_bytes_after_cleanup = 0`

### Import 幂等

重复导入同一文档：

```bash
curl -X POST http://127.0.0.1:8000/knowledge/import \
  -H 'Content-Type: application/json' \
  -d '{"title":"demo","source":"demo.md","source_type":"md","content":"# Demo\n\nsame content"}'
```

第二次预期：

```json
{
  "indexed_to_sqlite": false,
  "indexed_to_chroma": false,
  "skipped_reason": "content_unchanged"
}
```

### Health 检查

```bash
curl http://127.0.0.1:8000/knowledge/health
```

预期：

- SQLite / FTS / Chroma count 一致。
- 如果 segment 文件异常，出现 `chroma_segment_size_anomaly` warning。

## 面试表达版本

可以这样讲：

> 我在做 RAG eval 的时候遇到过一个比较典型的向量库工程问题：本地 Chroma 的 HNSW segment 文件被 eval 跑到了 250GB 左右，导致磁盘几乎打满。这个问题一开始不是通过 answer quality 发现的，而是 eval 跑不动、health count 又看起来基本正常，所以我开始排查 Chroma 的持久化目录、SQLite catalog、Chroma queue 和 segment 文件大小。

> 最后定位到根因是：eval 每次都会导入同一批 fixture 文档，而导入逻辑对同一个 `doc_id` 无条件执行 delete + upsert。对 HNSW 向量索引来说，这不是普通数据库的原地覆盖，delete 可能只是逻辑删除，upsert 会追加新向量，segment 扩容后也不会自动 compaction。再叠加 embedding 维度比较高，反复 churn 后 segment 文件持续膨胀。

> 我做了三层修复。第一层是 eval 隔离，把每次 eval 的 Chroma persist dir 切到独立目录，成功后自动清理，失败时通过 `--keep-chroma` 保留现场，并把 cleanup 前后大小写进 manifest。第二层是导入幂等，导入前计算 normalized content hash；只有内容和索引相关元数据都没变时，直接返回 `skipped_reason=content_unchanged`，不再触发 delete/upsert。第三层是 health 监控，不只看 chunk count，还要检查 Chroma segment 总大小、最大单文件和平均每 chunk 占用，发现异常时给出 warning。

> 这个问题对我的启发是：RAG 系统里向量库不是简单缓存，它有自己的存储模型和 compaction 行为。工程上不能只看召回率和答案质量，还要关注索引生命周期、导入幂等、磁盘水位、eval 环境隔离和可观测性，否则离线评估本身也可能成为破坏生产索引的来源。

## 面试加分点

- 这个问题体现了 RAG 工程里“索引生命周期管理”的重要性，不只是调用向量库 API。
- delete/upsert 在 HNSW 中可能导致磁盘膨胀，说明需要理解底层存储行为。
- eval 要和真实知识库隔离，否则测试会污染长期索引。
- content_hash 幂等不仅要看正文，也要看 title/source/metadata，因为 metadata 可能承载 ACL 或标签。
- health 不应该只检查 count，还要检查 segment size、queue backlog、orphan/missing chunks。
- 修复不是一次性清理磁盘，而是建立“预防 + 监控 + 可恢复”的闭环。
