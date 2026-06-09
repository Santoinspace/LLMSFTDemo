"""
ChromaDB 向量存储封装

提供文档的向量化存储和相似度检索能力。
- 使用 BAAI/bge-m3 生成嵌入
- ChromaDB 本地持久化
- 支持 metadata 过滤
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from chromadb import PersistentClient, Settings
from chromadb.utils import embedding_functions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "BAAI/bge-m3",
    ):
        """
        初始化向量存储

        参数:
            persist_directory: 持久化目录
            collection_name: 集合名称
            embedding_model: embedding 模型名称
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 确保持久化目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.client = PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # 创建 embedding 函数
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device="cuda" if self._cuda_available() else "cpu",
        )

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"向量存储初始化完成: {persist_directory}/{collection_name}")
        logger.info(f"当前文档数: {self.collection.count()}")

    @staticmethod
    def _cuda_available() -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        添加文档到向量存储

        参数:
            documents: 文档文本列表
            metadatas: 文档元数据列表
            ids: 文档 ID 列表（默认自动生成）
        """
        n = len(documents)

        if ids is None:
            # 使用时间戳 + 序号作为 ID
            import time
            ts = str(int(time.time() * 1000))
            ids = [f"doc_{ts}_{i}" for i in range(n)]

        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        # ChromaDB 一次添加大量文档可能较慢，分批处理
        batch_size = 100
        total_added = 0

        for i in range(0, n, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            total_added += len(batch_docs)

        logger.info(f"已添加 {total_added} 个文档")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_meta: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        相似度检索

        参数:
            query: 查询文本
            k: 返回的文档数量
            filter_meta: metadata 过滤条件

        返回:
            [{"content": str, "metadata": dict, "distance": float}, ...]
        """
        # 构建过滤条件
        where = None
        if filter_meta:
            where = filter_meta

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )

        # 格式化结果
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        formatted = []
        for i in range(len(documents)):
            formatted.append({
                "id": ids[i] if i < len(ids) else "",
                "content": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })

        return formatted

    def delete_collection(self) -> None:
        """删除当前 collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"已删除 collection: {self.collection_name}")
        # 重新创建空的 collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """返回文档数量"""
        return self.collection.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="向量存储管理")
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_db",
        help="持久化目录",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="集合名称",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["add", "search", "count", "delete"],
        required=True,
        help="操作: add/search/count/delete",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="搜索查询（--action=search 时使用）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="返回结果数量",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入文件路径（--action=add 时使用）",
    )

    args = parser.parse_args()

    store = VectorStore(
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
            print(f"[{i}] (distance={r['distance']:.4f}) {r['content'][:150]}...")
            print()

    elif args.action == "delete":
        confirm = input("确认删除 collection？(yes/no): ")
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
