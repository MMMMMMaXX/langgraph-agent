"""Lexical retrieval package.

这里不要 re-export factory，避免 KnowledgeCatalog 引入 tokenizer 时触发循环导入。
业务代码请直接从 `app.retrieval.lexical.factory` 引入 get_lexical_retriever。
"""
