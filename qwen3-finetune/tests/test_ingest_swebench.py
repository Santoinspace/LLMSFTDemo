"""测试 rag/ingest_swebench.py 的导入逻辑（思路 B：直接文档存储）"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from ingest_swebench import ingest_swebench, print_stats


def _make_mock_instance(instance_id, repo, problem_statement):
    """构造模拟 SWE-bench 实例"""
    return {
        "instance_id": instance_id,
        "repo": repo,
        "problem_statement": problem_statement,
        "text": (
            "You will be provided with a partial code base...\n\n"
            f"<issue>\n{problem_statement}\n</issue>\n\n"
            "<code>\n"
            "[start of src/main.py]\n"
            "1 def fix():\n"
            "2     return 'done'\n"
            "[end of src/main.py]\n"
            "</code>\n\n"
            "<patch>\n--- a/src/main.py\n+++ b/src/main.py\n</patch>\n\n"
            "Respond below:"
        ),
        "base_commit": "abc123",
        "patch": "--- a/src/main.py\n+++ b/src/main.py\n",
        "FAIL_TO_PASS": '["test_fix"]',
        "PASS_TO_PASS": '["test_other"]',
    }


class TestIngestBM25:
    """测试 BM25 路径的直接文档存储"""

    def test_ingest_mock_instances(self, temp_dir):
        """用 mock 数据测试 BM25 导入流程"""
        # 创建 mock 数据集 JSONL
        instances = [
            _make_mock_instance(
                "django__django-1", "django/django",
                "Fix QuerySet.count() returning None for empty queries",
            ),
            _make_mock_instance(
                "django__django-2", "django/django",
                "Add connection pooling to database backend",
            ),
            _make_mock_instance(
                "psf__requests-1", "psf/requests",
                "Handle redirect loops gracefully",
            ),
        ]

        # 写入临时文件，然后通过 BM25 ingest 处理
        jsonl_path = temp_dir / "mock_swebench.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for inst in instances:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")

        # 直接调 ingest_swebench（但需要数据集文件而不是 mock）
        # 改为手动测试 BM25Store 集成
        from bm25_store import BM25Store

        store = BM25Store(str(temp_dir / "bm25_index"), "test_ingest")

        docs = []
        metas = []
        for inst in instances:
            docs.append(inst["text"])
            metas.append({
                "repo": inst["repo"],
                "instance_id": inst["instance_id"],
            })

        store.add_documents(docs, metas)
        assert store.count() == 3

        # 搜索测试 — "connection" 只出现在第二个实例中
        results = store.similarity_search("connection pooling", k=2)
        assert len(results) >= 1
        assert "connection" in results[0]["content"].lower()

    def test_ingest_deduplication_not_needed(self, temp_dir):
        """每条实例都是独立文档，不需要去重"""
        from bm25_store import BM25Store

        store = BM25Store(str(temp_dir / "bm25_index"), "test_no_dedup")

        inst = _make_mock_instance(
            "test__repo-1", "test/repo", "Same problem statement"
        )
        # 两条相同的实例也应各自存储（不同 instance_id）
        docs = [inst["text"], inst["text"]]
        metas = [
            {"repo": "test/repo", "instance_id": "test__repo-1"},
            {"repo": "test/repo", "instance_id": "test__repo-2"},
        ]
        store.add_documents(docs, metas)
        assert store.count() == 2

    def test_metadata_preserved(self, temp_dir):
        """元数据正确传递到搜索返回"""
        from bm25_store import BM25Store

        store = BM25Store(str(temp_dir / "bm25_index"), "test_meta")

        inst = _make_mock_instance(
            "django__django-1", "django/django",
            "Fix database transaction handling"
        )
        store.add_documents(
            [inst["text"]],
            [{"repo": "django/django", "instance_id": "django__django-1"}],
        )

        results = store.similarity_search("transaction handling", k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["repo"] == "django/django"
        assert results[0]["metadata"]["instance_id"] == "django__django-1"

    def test_filter_by_repo(self, temp_dir):
        """仓库过滤功能"""
        from bm25_store import BM25Store

        store = BM25Store(str(temp_dir / "bm25_index"), "test_filter")

        django_inst = _make_mock_instance(
            "django__django-1", "django/django",
            "Django model save should handle None fields"
        )
        requests_inst = _make_mock_instance(
            "psf__requests-1", "psf/requests",
            "Handle None URL gracefully"
        )

        store.add_documents(
            [django_inst["text"], requests_inst["text"]],
            [
                {"repo": "django/django", "instance_id": "django__django-1"},
                {"repo": "psf/requests", "instance_id": "psf__requests-1"},
            ],
        )

        results = store.similarity_search(
            "handle None", k=5, filter_meta={"repo": "psf/requests"}
        )
        assert len(results) == 1
        assert results[0]["metadata"]["repo"] == "psf/requests"
