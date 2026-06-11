"""
SWE-bench_bm25_27K 知识库导入脚本

将 SWE-bench 实例直接存为可 BM25 检索的文档。
每条实例 = problem_statement（搜索文本） + text（完整上下文，含 BM25 已检索的代码文件 + patch）。

索引: 纯 CPU，秒级完成，无需 GPU。

使用方式:
    python rag/ingest_swebench.py --max_instances 500 --shuffle
    python rag/ingest_swebench.py --stats
    python rag/ingest_swebench.py --retriever chromadb --max_instances 100
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_stats(split: str = "train", max_instances: Optional[int] = None):
    """打印数据集统计信息"""
    from datasets import load_dataset

    logger.info("加载数据集...")
    dataset = load_dataset(
        "princeton-nlp/SWE-bench_bm25_27K",
        split=split,
        streaming=True,
    )

    repo_counter = Counter()
    total = 0

    for instance in dataset:
        repo = instance.get("repo", "unknown")
        repo_counter[repo] += 1
        total += 1

        if total % 500 == 0:
            logger.info(f"已扫描 {total} 条...")
        if max_instances and total >= max_instances:
            break

    print(f"\n{'='*60}")
    print(f" SWE-bench_bm25_27K ({split}) 统计")
    print(f"{'='*60}")
    print(f"  总实例数: {total}")
    print(f"  仓库数:   {len(repo_counter)}")
    print(f"\n  按仓库分布 (Top 20):")
    print(f"  {'Repo':<40} {'Count':>8}")
    print(f"  {'-'*48}")
    for repo, count in repo_counter.most_common(20):
        print(f"  {repo:<40} {count:>8}")


def ingest_swebench(
    split: str = "train",
    max_instances: Optional[int] = None,
    filter_repo: Optional[str] = None,
    shuffle: bool = False,
    seed: int = 42,
    persist_dir: str = "./bm25_index_swebench",
    collection_name: str = "swebench_instances",
    retriever: str = "bm25",
) -> Dict:
    """主导入流程：stream 数据集 → 每条实例存为完整文档 → BM25/ChromaDB 索引"""

    from datasets import load_dataset

    logger.info("=" * 60)
    logger.info(" SWE-bench 知识库导入")
    logger.info("=" * 60)
    logger.info(f"Split: {split}, Retriever: {retriever}")
    logger.info(f"输出: {persist_dir}/{collection_name}")

    # 初始化检索存储
    if retriever == "bm25":
        from bm25_store import BM25Store
        store = BM25Store(
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
    elif retriever == "chromadb":
        from vector_store import VectorStore
        store = VectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"未知检索器: {retriever}")

    stats = {"instances_processed": 0, "documents_added": 0}

    try:
        logger.info("加载数据集 (streaming)...")
        dataset = load_dataset(
            "princeton-nlp/SWE-bench_bm25_27K",
            split=split,
            streaming=True,
        )
        if shuffle:
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)

        # 批量收集，定期 flush
        batch_docs = []
        batch_metas = []
        batch_size = 200

        for instance in dataset:
            repo = instance.get("repo", "")
            if filter_repo and repo != filter_repo:
                continue

            instance_id = instance.get("instance_id", "")
            problem_statement = instance.get("problem_statement", "")
            text = instance.get("text", "")

            if not problem_statement or not text:
                continue

            searchable_text = problem_statement
            stored_text = text

            batch_docs.append(stored_text)
            batch_metas.append({
                "repo": repo,
                "instance_id": instance_id,
                "file_path": f"swebench::{repo}::{instance_id}",
            })
            stats["documents_added"] += 1

            if len(batch_docs) >= batch_size:
                store.add_documents(batch_docs, batch_metas)
                batch_docs.clear()
                batch_metas.clear()
                logger.info(
                    f"已添加 {stats['documents_added']} 文档 "
                    f"({stats['instances_processed']} 实例)"
                )

            stats["instances_processed"] += 1
            if stats["instances_processed"] % 500 == 0:
                logger.info(f"进度: {stats['instances_processed']} 实例")

            if max_instances and stats["instances_processed"] >= max_instances:
                break

        # 最终 flush
        if batch_docs:
            store.add_documents(batch_docs, batch_metas)

    except Exception:
        logger.exception("导入过程出错")
        raise

    print(f"\n{'='*60}")
    print(f" 导入完成")
    print(f"{'='*60}")
    print(f"  处理实例:  {stats['instances_processed']}")
    print(f"  添加文档:  {stats['documents_added']}")
    print(f"  索引总数:  {store.count()}")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench 知识库导入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rag/ingest_swebench.py --max_instances 500 --shuffle
  python rag/ingest_swebench.py --retriever chromadb --max_instances 100
  python rag/ingest_swebench.py --stats
  python rag/ingest_swebench.py --stats --max_instances 200
        """,
    )
    parser.add_argument("--split", type=str, default="train",
                        help="数据集 split (默认: train)")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="最大处理实例数")
    parser.add_argument("--filter_repo", type=str, default=None,
                        help="只处理指定仓库")
    parser.add_argument("--shuffle", action="store_true",
                        help="随机打乱")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--retriever", type=str, default="bm25",
                        choices=["bm25", "chromadb"],
                        help="检索器类型 (默认: bm25)")
    parser.add_argument("--persist_dir", type=str, default=None,
                        help="索引持久化目录")
    parser.add_argument("--collection", type=str, default="swebench_instances",
                        help="集合名称")
    parser.add_argument("--stats", action="store_true",
                        help="只打印统计信息，不导入")

    args = parser.parse_args()

    if args.stats:
        print_stats(split=args.split, max_instances=args.max_instances)
        return

    # 默认 persist_dir 根据 retriever 自动选择
    if args.persist_dir is None:
        args.persist_dir = (
            "./bm25_index_swebench"
            if args.retriever == "bm25"
            else "./chroma_db_swebench"
        )

    ingest_swebench(
        split=args.split,
        max_instances=args.max_instances,
        filter_repo=args.filter_repo,
        shuffle=args.shuffle,
        seed=args.seed,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        retriever=args.retriever,
    )


if __name__ == "__main__":
    main()
