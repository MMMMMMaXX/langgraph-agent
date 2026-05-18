"""可观测性公共模块。

PR-0 落地：
- `redactor`：统一 prompt / chunk / API key / token 脱敏接口；
- `emit`：fire-and-forget 埋点入口，封装基数 / 白名单 / 自监控逻辑。

后续 PR 在埋点时只能从本模块导入 `emit_counter / emit_histogram / emit_gauge`，
不允许业务代码直接调用 prometheus_client。
"""
