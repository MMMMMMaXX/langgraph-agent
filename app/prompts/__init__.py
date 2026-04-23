"""集中管理项目里的 prompt。

第一版目标不是做复杂模板系统，而是先解决两个工程问题：
1. prompt 不再散落在各 agent / tool / util 文件里
2. 后续改 prompt 时，可以先在 prompts 目录里统一查找和评审

当前策略：
- 固定 system prompt：定义为常量
- 少量动态 prompt：定义为函数，避免在业务代码里手写长字符串
"""

