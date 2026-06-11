"""
Whoosh BM25 检索存储

提供与 VectorStore 一致的接口，基于 Whoosh 实现 BM25 全文检索。
纯 CPU，秒级索引，毫秒级检索，无需 GPU。

持久化：索引存储在 <persist_dir>/<collection_name>/ 目录。
"""
import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from whoosh import index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 默认 Whoosh schema
def _default_schema() -> Schema:
    return Schema(
        id=ID(stored=True, unique=True),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        repo=ID(stored=True),
        instance_id=ID(stored=True),
        file_path=ID(stored=True),
        file_type=ID(stored=True),
        chunk_index=NUMERIC(stored=True),
        chunk_size=NUMERIC(stored=True),
    )


class BM25Store:
    """Whoosh BM25 全文检索存储"""

    def __init__(
        self,
        persist_directory: str = "./bm25_index",
        collection_name: str = "documents",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.index_dir = Path(persist_directory) / collection_name

        if self.index_dir.exists():
            self._idx = index.open_dir(str(self.index_dir))
            logger.info(
                f"已打开索引: {self.index_dir} (文档数: {self._idx.doc_count()})"
            )
        else:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._idx = index.create_in(str(self.index_dir), _default_schema())
            logger.info(f"已创建索引: {self.index_dir}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """批量添加文档到索引"""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if ids is None:
            ts = str(int(time.time() * 1000))
            ids = [f"doc_{ts}_{i}" for i in range(len(documents))]

        writer = self._idx.writer()
        added = 0
        for doc_id, content, meta in zip(ids, documents, metadatas):
            if not content or not content.strip():
                continue
            writer.update_document(
                id=doc_id,
                content=content,
                repo=meta.get("repo", ""),
                instance_id=meta.get("instance_id", ""),
                file_path=meta.get("file_path", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                chunk_size=meta.get("chunk_size", 0),
            )
            added += 1

        writer.commit()
        logger.info(f"已添加 {added} 个文档")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_meta: Optional[Dict] = None,
    ) -> List[Dict]:
        """BM25 全文检索，返回 top-k 结果"""
        if not query or not query.strip():
            return []

        results_list = []
        with self._idx.searcher() as searcher:
            parser = MultifieldParser(
                ["content", "repo"], self._idx.schema
            )
            q = parser.parse(query)

            # metadata 过滤
            if filter_meta:
                from whoosh.query import And, Term

                terms = [
                    Term(k, str(v)) for k, v in filter_meta.items() if v
                ]
                if terms:
                    q = And([q, And(terms)])

            hits = searcher.search(q, limit=k)

            for hit in hits:
                results_list.append({
                    "id": hit.get("id", ""),
                    "content": hit.get("content", ""),
                    "metadata": {
                        "repo": hit.get("repo", ""),
                        "instance_id": hit.get("instance_id", ""),
                        "file_path": hit.get("file_path", ""),
                        "file_type": hit.get("file_type", ""),
                        "chunk_index": hit.get("chunk_index", 0),
                        "chunk_size": hit.get("chunk_size", 0),
                    },
                    "distance": hit.score,
                })

        return results_list

    def delete_collection(self) -> None:
        """删除整个索引目录并重建空索引"""
        if self.index_dir.exists():
            shutil.rmtree(str(self.index_dir))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._idx = index.create_in(str(self.index_dir), _default_schema())
        logger.info(f"已删除并重建索引: {self.index_dir}")

    def count(self) -> int:
        """返回文档数量"""
        return self._idx.doc_count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 检索存储管理")
    parser.add_argument("--persist_dir", type=str, default="./bm25_index",
                        help="持久化目录")
    parser.add_argument("--collection", type=str, default="documents",
                        help="集合名称")
    parser.add_argument("--action", type=str, choices=["add", "search", "count", "delete"],
                        required=True, help="操作: add/search/count/delete")
    parser.add_argument("--query", type=str, default="",
                        help="搜索查询（--action=search 时使用）")
    parser.add_argument("--k", type=int, default=5,
                        help="返回结果数量")
    parser.add_argument("--input", type=str, default=None,
                        help="输入 JSONL 文件路径（--action=add 时使用）")

    args = parser.parse_args()

    store = BM25Store(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
    )

    if args.action == "count":
        print(f"文档数量: {store.count()}")

    elif args.action == "search":
        if not args.query:
            print("请使用 --query 指定搜索内容")
            exit(1)
        results = store.similarity_search(args.query, k=args.k)
        print(f"\n搜索: {args.query}")
        print("=" * 60)
        for i, r in enumerate(results, 1):
            print(f"[{i}] (score={r['distance']:.4f}) {r['content'][:200]}...")
            print()

    elif args.action == "delete":
        confirm = input("确认删除索引？(yes/no): ")
        if confirm.lower() == "yes":
            store.delete_collection()
            print("已删除")

    elif args.action == "add":
        if not args.input:
            print("请使用 --input 指定输入文件路径")
            exit(1)

        import json
        documents = []
        metadatas = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                documents.append(item.get("text", item.get("content", "")))
                metadatas.append(item.get("metadata", {}))

        store.add_documents(documents, metadatas)
        print(f"已添加 {len(documents)} 个文档")
